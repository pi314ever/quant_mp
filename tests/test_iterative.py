import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from quant_mp.algs.template import get_algorithm
from quant_mp.datatypes.template import get_data_format
from quant_mp.quantizer import quant


def _prepare_inputs(B: int, N: int, device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    return x


@pytest.mark.parametrize("bit_width", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_iterative_fit_params_single_iter(bit_width: int, symmetric: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 64
    x = _prepare_inputs(B, N, device)

    # Setup
    df = get_data_format(f"int{bit_width}")
    alg = get_algorithm("iterative", algorithm_init_kwargs={"num_iters": 1})

    init_scale = torch.full((B, 1), 0.3, device=device)
    init_shift = None if symmetric else torch.full((B, 1), -0.1, device=device)

    # Algorithm output (one iteration)
    scale, shift = alg.fit_params(
        df, x, init_scale.clone(), None if symmetric else init_shift.clone()
    )

    # Reference: quantize once with initial params, then apply normal-equation updates
    x_quant, _ = quant(df, x, init_scale, None if symmetric else init_shift)
    if symmetric:
        num = torch.sum(x * x_quant, dim=1, keepdim=True)
        den = torch.sum(x_quant**2, dim=1, keepdim=True)
        scale_ref = num / den
        shift_ref = None
    else:
        num = torch.sum((x - init_shift) * x_quant, dim=1, keepdim=True)
        den = torch.sum(x_quant**2, dim=1, keepdim=True)
        scale_ref = num / den
        # Match implementation detail: divides by batch size (len(input)), not feature count
        shift_ref = torch.sum(x - scale_ref * x_quant, dim=1, keepdim=True) / x.shape[0]

    # Compare
    assert scale.shape == init_scale.shape
    assert torch.allclose(scale, scale_ref, rtol=1e-4, atol=1e-5)
    if symmetric:
        assert shift is None
    else:
        assert shift is not None and shift.shape == init_shift.shape
        assert torch.allclose(shift, shift_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("world_size", [2])
@pytest.mark.parametrize("num_iters", [1, 3])
def test_iterative_fit_params_distributed(world_size: int, num_iters: int):
    """Verify distributed reductions align with manual global reference."""
    port = _get_free_port()
    mp.spawn(
        _iterative_worker,
        args=(world_size, port, num_iters),
        nprocs=world_size,
        join=True,
    )


def _iterative_worker(rank: int, world_size: int, port: int, num_iters: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = torch.device("cpu")
    torch.manual_seed(1234 + rank)
    B, N = 3, 8
    x_local = torch.randn(B, N, device=device)

    df = get_data_format("int4")
    alg = get_algorithm("iterative", algorithm_init_kwargs={"num_iters": num_iters})
    init_scale = torch.full((B, 1), 0.3, device=device)

    scale, shift = alg.fit_params(df, x_local, init_scale.clone(), None)
    assert shift is None

    gathered_x = [torch.empty_like(x_local) for _ in range(world_size)]
    dist.all_gather(gathered_x, x_local)

    gathered_scale = [torch.empty_like(scale) for _ in range(world_size)]
    dist.all_gather(gathered_scale, scale)
    for other in gathered_scale[1:]:
        torch.testing.assert_close(scale, other, rtol=1e-6, atol=1e-7)

    ref_scale = torch.empty_like(scale)
    if rank == 0:
        ref_scale.copy_(
            _iterative_reference(df, init_scale.cpu(), [t.cpu() for t in gathered_x], num_iters)
        )
    dist.broadcast(ref_scale, src=0)

    torch.testing.assert_close(scale, ref_scale.to(device), rtol=1e-5, atol=1e-6)

    dist.barrier()
    dist.destroy_process_group()


def _iterative_reference(df, init_scale, tensors, num_iters):
    scale = init_scale.clone()
    for _ in range(num_iters):
        sum_x = torch.zeros_like(scale)
        sum_x_q2 = torch.zeros_like(scale)
        for chunk in tensors:
            x_quant, _ = quant(df, chunk, scale, None)
            sum_x += torch.sum(chunk * x_quant, dim=1, keepdim=True)
            sum_x_q2 += torch.sum(x_quant**2, dim=1, keepdim=True)
        scale = sum_x / sum_x_q2
    return scale


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("bit_width", [2, 4, 8])
def test_iterative_fit_params_all_zero_quantization(bit_width: int):
    """
    When x_quant is identically zero (e.g. because the initial scale is extremely large),
    the denominator sum_x_quant_sq vanishes and the current implementation silently
    produces NaN scales. This test documents the regression.
    """
    device = torch.device("cpu")
    df = get_data_format(f"int{bit_width}")
    alg = get_algorithm("iterative", algorithm_init_kwargs={"num_iters": 1})
    B, N = 2, 8

    # All activations are tiny compared to the initial scale so they quantize to zero.
    init_scale = torch.full((B, 1), 1e6, device=device)
    inputs = torch.full((B, N), 1e-2, device=device)

    x_quant, _ = quant(df, inputs, init_scale, None)
    assert torch.count_nonzero(x_quant) == 0, "setup must produce zero-valued quantization"

    scale, shift = alg.fit_params(df, inputs, init_scale.clone(), None)
    assert shift is None
    assert torch.isfinite(scale).all(), "Iterative.fit_params should guard against zero denominators"


def test_iterative_fit_params_zero_quantization_with_shift():
    """
    The asymmetric pathway uses the same denominator, so NaNs also propagate to the learned shift.
    """
    device = torch.device("cpu")
    df = get_data_format("int4")
    alg = get_algorithm("iterative", algorithm_init_kwargs={"num_iters": 2})
    B, N = 3, 6

    init_scale = torch.full((B, 1), 5e4, device=device)
    init_shift = torch.full((B, 1), 0.5, device=device)

    # Craft inputs that exactly equal the shift so the quantized tensor is zero.
    inputs = torch.full((B, N), 0.5, device=device)
    x_quant, _ = quant(df, inputs, init_scale, init_shift)
    assert torch.count_nonzero(x_quant) == 0, "setup must produce zero-valued quantization"

    scale, shift = alg.fit_params(df, inputs, init_scale.clone(), init_shift.clone())
    assert shift is not None
    assert torch.isfinite(scale).all(), "Scale should remain finite even if quantized tensor is zero"
    assert torch.isfinite(shift).all(), "Shift should remain finite even if quantized tensor is zero"
