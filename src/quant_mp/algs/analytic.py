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
        from quant_mp.datatypes import FloatDataFormat, UniformDataFormat

        # TODO: Generalize axis if needed
        param_shape = scale.shape
        if dist.is_initialized():
            mean, x_std = dist_std(input, dim=1, dtype=input.dtype)
        else:
            mean = torch.mean(input, dim=1).reshape(param_shape)
            x_std = torch.std(input, dim=1)
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
        return scale.reshape(param_shape), shift

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
        return self.ste(ctx, quant_mask, grad_output)


def dist_std(
    x: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    unbiased: bool = False,
    dtype=torch.float32,
):
    x_acc = x.detach().to(dtype)

    local_sum = x_acc.sum(dim=dim, keepdim=True).contiguous()
    local_sumsq = (x_acc * x_acc).sum(dim=dim, keepdim=True).contiguous()

    per_rank_count = torch.tensor(
        x_acc.shape[dim], device=x.device, dtype=torch.float64
    )
    local_count = per_rank_count.expand_as(local_sum).contiguous()

    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_sumsq, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

    N = local_count  # already float64, same shape as local_sum
    mean = local_sum / N.clamp_min(1)
    ex2 = local_sumsq / N.clamp_min(1)
    var = (ex2 - mean * mean).clamp_min(0)

    if unbiased:
        # sample variance = var * N/(N-1) elementwise, guard N>1
        var = torch.where(N > 1, var * (N / (N - 1)), torch.zeros_like(var))

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
