import torch

from quant_mp.algs.template import get_algorithm
from quant_mp.config import QuantConfig
from quant_mp.datatypes.template import get_data_format
from quant_mp.QModules import QuantFunction


def _get_octav_config(df_name: str):
    df = get_data_format(df_name)
    alg = get_algorithm("octav", algorithm_init_kwargs={"num_iters": 1})
    qcfg = QuantConfig(
        qval_data_format=df,
        qparam_data_format=df,
        algorithm=alg,
        symmetric=True,
    )
    return df, alg, qcfg


def test_octav_fit_params_fp16_overflow():
    """
    Octav accumulates statistics in the dtype of `scale`. When that dtype is fp16,
    large tensors overflow to inf and permanently corrupt the learned scale.
    """
    device = torch.device("cpu")
    df, alg, _ = _get_octav_config("int4")
    B, N = 2, 8192  # 8192 * 10 > fp16 max => overflow
    init_scale = torch.ones((B, 1), dtype=torch.float16, device=device)
    inputs = torch.full((B, N), 10.0, dtype=torch.float16, device=device)

    scale, shift = alg.fit_params(df, inputs, init_scale.clone(), None)

    assert shift is None
    assert torch.isfinite(scale).all(), (
        "Octav.fit_params should guard fp16 reductions from overflowing to inf"
    )
    assert scale.dtype == init_scale.dtype


def test_octav_backward_handles_infinite_scale_masking():
    """
    The backward correction should not introduce NaNs even when provided an infinite scale tensor,
    as long as all elements are inside the quantization range (outside_mask is False everywhere).
    """
    device = torch.device("cpu")
    _, _, qcfg = _get_octav_config("int4")

    x = torch.tensor(
        [[0.1, 0.2]], dtype=torch.float32, device=device, requires_grad=True
    )
    inf_scale = torch.tensor([[float("inf")]], dtype=torch.float32, device=device)
    y = QuantFunction.apply(x, inf_scale, None, qcfg)
    y.sum().backward()

    assert torch.isfinite(x.grad).all()
