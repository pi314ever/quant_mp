import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from quant_mp.algs.analytic import dist_std, get_copt_uniform
from quant_mp.algs.template import get_algorithm
from quant_mp.datatypes.template import get_data_format


def _prepare_inputs(B: int, N: int, device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    return x


@pytest.mark.parametrize("bit_width", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_analytic_fit_params_uniform(bit_width: int, symmetric: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 64
    x = _prepare_inputs(B, N, device)

    # Setup
    df = get_data_format(f"int{bit_width}")
    alg = get_algorithm("analytic")
    init_scale = torch.ones(B, 1, device=device)
    init_shift = None if symmetric else torch.zeros(B, 1, device=device)

    # Algorithm output
    scale, shift = alg.fit_params(df, x, init_scale, init_shift)

    # Reference calculation
    x_std = torch.std(x, dim=1, keepdim=True)
    copt = get_copt_uniform(df)
    scale_ref = (2.0 * copt * x_std) / (df.n_values - 1)
    shift_ref = None if symmetric else x.mean(dim=1, keepdim=True)

    # Compare
    assert scale.shape == init_scale.shape
    assert torch.allclose(scale, scale_ref.reshape_as(scale), rtol=1e-4, atol=1e-5)
    if symmetric:
        assert shift is None
    else:
        assert shift is not None and shift.shape == init_shift.shape
        assert torch.allclose(shift, shift_ref.reshape_as(shift), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("world_size", [2])
def test_distributed_std_dim_cpu(world_size):
    """Runs on CPU with gloo backend."""
    port = _get_free_port()
    mp.spawn(
        _worker, args=(world_size, "gloo", port, "cpu"), nprocs=world_size, join=True
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Needs at least 2 CUDA devices",
)
@pytest.mark.parametrize("world_size", [2])
def test_distributed_std_dim_cuda(world_size):
    """Runs on GPU with nccl backend (skipped if <2 GPUs)."""
    port = _get_free_port()
    mp.spawn(
        _worker, args=(world_size, "nccl", port, "cuda"), nprocs=world_size, join=True
    )


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, backend: str, port: int, device_type: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    # Make each rank's tensor different but deterministic
    torch.manual_seed(1234 + rank)
    x = torch.randn(4, 3, device=device)  # [rows, cols]

    # Compute distributed mean/std along dim=0 (i.e., across rows)
    mean, std = dist_std(x, dim=0, unbiased=False)

    # Build reference by all_gathering full tensors and computing local mean/std on rank 0
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    if rank == 0:
        full = torch.cat(gathered, dim=0)  # shape = [4*world_size, 3]

        ref_mean = full.mean(dim=0)
        ref_std = full.std(dim=0, unbiased=False)

        torch.testing.assert_close(mean, ref_mean, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(std, ref_std, rtol=1e-5, atol=1e-6)

    dist.barrier()
    dist.destroy_process_group()
