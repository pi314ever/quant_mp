import math
from functools import cache
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes import FloatDataFormat, UniformDataFormat
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Analytic(Algorithm):
    name = "analytic"
    has_custom_gradients = True
    has_fit_params = True

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,  # pyright: ignore[reportDeprecated]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Fit scale/shift analytically from input statistics.

        Args:
            data_format: Target data format for quantized values.
            input: Block-flattened tensor shaped ``[num_blocks, block_size]``.
            scale: Scale tensor shaped ``[num_blocks, 1]`` to update.
            shift: Optional shift tensor shaped ``[num_blocks, 1]``; ``None`` when symmetric.
        """
        from quant_mp.datatypes import FloatDataFormat, UniformDataFormat

        # TODO: Generalize axis if needed
        param_shape = scale.shape
        orig_dtype = scale.dtype
        input_acc = input.to(dtype=torch.float32)
        if dist.is_initialized():
            mean, x_std = dist_std(
                input_acc, dim=1, keepdim=True, unbiased=False, dtype=torch.float32
            )
        else:
            mean = torch.mean(input_acc, dim=1, keepdim=True)
            x_std = torch.std(input_acc, dim=1, keepdim=True)
        if isinstance(data_format, UniformDataFormat):
            scale = (2 * get_copt_uniform(data_format) * x_std) / (
                data_format.n_values - 1
            )
        elif isinstance(data_format, FloatDataFormat):  # Float Data Format
            scale = get_copt_float(data_format) * x_std / data_format.max_value
        else:
            scale = get_copt_general(data_format) * x_std / data_format.max_value
        if shift is not None:
            shift = mean
        return scale.reshape(param_shape).to(
            dtype=orig_dtype
        ), None if shift is None else shift.to(dtype=orig_dtype)

    def compute_gradients(
        self,
        ctx,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        quant_mask: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Use STE gradients for analytic quantization.

        Args:
            ctx: Autograd context.
            data_format: Data format used during quantization.
            input: Block-flattened tensor shaped ``[num_blocks, block_size]``.
            scale: Scale tensor shaped ``[num_blocks, 1]``.
            shift: Optional shift tensor shaped ``[num_blocks, 1]`` or ``None``.
            quant_mask: Mask tensor shaped like ``input`` indicating in-range values.
            grad_output: Upstream gradient shaped like ``input``.
        """
        return self.ste(ctx, quant_mask, grad_output)


def dist_std(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    unbiased: bool = False,
    dtype=torch.float32,
):
    dim = dim % x.dim()
    x_acc = x.detach().to(dtype)
    local_mean = torch.mean(x_acc, dim=dim, keepdim=True)
    centered = x_acc - local_mean
    local_m2 = torch.sum(centered * centered, dim=dim, keepdim=True)

    count = (
        torch.tensor(
            x_acc.shape[dim],
            device=x.device,
            dtype=torch.float64,
        )
        .reshape([1] * x_acc.dim())
        .expand_as(local_mean)
    )

    local_mean64 = local_mean.to(torch.float64)
    local_m264 = local_m2.to(torch.float64)
    local_sum = local_mean64 * count
    local_sumsq = local_m264 + (local_mean64**2) * count

    global_stats = torch.stack((local_sum, local_sumsq, count), dim=0)
    if dist.is_initialized():
        dist.all_reduce(global_stats, op=dist.ReduceOp.SUM)

    sum_all, sumsq_all, count_all = global_stats[0], global_stats[1], global_stats[2]
    denom = count_all.clamp_min(1.0)
    mean = sum_all / denom
    ex2 = sumsq_all / denom
    var = (ex2 - mean * mean).clamp_min(0.0)
    if unbiased:
        mask = count_all > 1
        adj = torch.where(mask, count_all - 1.0, torch.ones_like(count_all))
        var = torch.where(mask, (count_all / adj) * var, torch.zeros_like(var))

    if not keepdim:
        mean = mean.squeeze(dim)
        var = var.squeeze(dim)

    return mean.to(x.dtype), torch.sqrt(var).to(x.dtype)


def error(x, xdeq):
    err = torch.sum(((x - xdeq) ** 2)) / len(x)
    return err


def q_function(x):
    """Gaussian Q-function using torch/math erf depending on input type."""
    if isinstance(x, torch.Tensor):
        return 0.5 - 0.5 * torch.erf(x / math.sqrt(2.0))
    # scalar path
    return 0.5 - 0.5 * math.erf(x / math.sqrt(2.0))


def gauss_cdf(x, m, std):
    return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.0)) * std)))


def snr_float(G, xr, vr, C, sigma2):
    Cmax = G[-1]
    s = C / Cmax
    sigma2 = torch.tensor(sigma2)

    res = 2 * (1 + C**2 / sigma2) * q_function(C / torch.sqrt(sigma2))
    res += (
        -C
        * torch.sqrt(torch.tensor(2.0) / torch.pi)
        * torch.exp(-0.5 * (C**2) / sigma2)
        / torch.sqrt(sigma2)
    )

    F = gauss_cdf(s[:, None] * xr[None], 0.0, torch.sqrt(sigma2))
    p = F[:, 1:] - F[:, :-1]

    res += torch.sum(((s[:, None] * vr[None]) ** 2) * p / (12 * sigma2), dim=1)
    return 1 / res


def snr_uniform(C, sigma2, N):
    if not isinstance(C, torch.Tensor):
        C = torch.tensor(C)

    z = C**2 / sigma2
    return 1 / (
        2 * (1 + z) * q_function(torch.sqrt(z))
        - torch.sqrt(2 * z / math.pi) * torch.exp(-0.5 * z)
        + z / (3 * ((N - 1) ** 2))
    )


def snr_general(dataformat, C, sigma2):
    grid = C.unsqueeze(1) * dataformat.get_representable_values() / dataformat.max_value
    data = torch.randn((10000,))

    err = []
    for g in grid:
        qdata = g[torch.argmin(torch.abs(data.unsqueeze(1) - g), dim=-1)]
        err.append(torch.mean((data - qdata) ** 2))

    return sigma2 / torch.tensor(err)


@cache
def get_copt_uniform(data_format: "UniformDataFormat") -> float:
    C = torch.linspace(1.0, 100.0, steps=10000)
    gres = snr_uniform(C, torch.tensor(1.0), data_format.n_values)
    idx = int(torch.argmax(gres).item())
    return float(C[idx].item())


@cache
def get_copt_float(data_format: "FloatDataFormat") -> float:
    C = torch.linspace(1.0, 100.0, steps=10000)
    xr, vr = data_format.compute_interval_step_size()
    gres = snr_float(data_format.get_representable_values(), xr, vr, C, 1.0)
    idx = int(torch.argmax(gres).item())
    return float(C[idx].item())


@cache
def get_copt_general(data_format: "DataFormat") -> float:
    C = torch.linspace(1.0, 100.0, steps=10000)
    gres = snr_general(data_format, C, 1.0)
    idx = int(torch.argmax(gres).item())
    return float(C[idx].item())
