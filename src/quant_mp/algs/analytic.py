from functools import cache
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy import special

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes import FloatDataFormat, UniformDataFormat
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Analytic(Algorithm):
    name = "analytic"
    has_custom_gradients = True
    has_update_params = True

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,  # pyright: ignore[reportDeprecated]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # NOTE: Old implementation. Maybe can consolidate into one impl?
        from quant_mp.datatypes import FloatDataFormat, UniformDataFormat

        if not isinstance(data_format, (UniformDataFormat, FloatDataFormat)):
            raise RuntimeError(f"Invalid data format: {data_format}")

        # TODO: Generalize axis if needed
        x_std = torch.std(input, axis=1)
        if isinstance(data_format, UniformDataFormat):
            scale = (2 * get_copt_uniform(data_format) * x_std) / (
                data_format.n_values - 1
            )
        else:  # Float Data Format
            scale = get_copt_float(data_format) * x_std / data_format.max_value
        if shift is not None:
            shift = torch.mean(input, axis=1)
        return scale, shift

    def compute_gradients(
        self,
        ctx,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        grad_output: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.ste(ctx, grad_output)


def error(x, xdeq):
    err = torch.sum(((x - xdeq) ** 2)) / len(x)
    return err


def q_function(x):
    return 0.5 - 0.5 * special.erf(x / np.sqrt(2))


def gauss_cdf(x, m, std):
    return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.0)) * std)))


def snr_float(G, xr, vr, C, sigma2):
    Cmax = G[-1]
    s = C / Cmax
    sigma2 = torch.tensor(sigma2)
    C = torch.tensor(C)

    res = 2 * (1 + C**2 / sigma2) * q_function(C / torch.sqrt(sigma2))
    res += (
        -C
        * torch.sqrt(torch.tensor(2.0) / torch.pi)
        * torch.exp(-0.5 * (C**2) / sigma2)
        / torch.sqrt(sigma2)
    )

    F = gauss_cdf(s[:, None] * xr[None], 0.0, torch.sqrt(sigma2))
    p = F[:, 1:] - F[:, :-1]

    res += torch.sum(((s[:, None] * vr[None]) ** 2) * p / (12 * sigma2), axis=1)
    return 1 / res

def compute_float_grid(data_format: "FloatDataFormat"):
    # Init quant floating point grid
    kmax = (
        2 ** (data_format.exponent + data_format.mantissa)
        - 1
        - data_format.correction_factor
    )
    R = kmax // 2**data_format.mantissa + (kmax % 2**data_format.mantissa > 0) * 1 - 1
    R = 2 * R - 1

    G = data_format.get_representable_values()

    vr = torch.tensor(
        [
            2 ** (abs(r - 1 - R // 2) + 1 - data_format.mantissa - data_format.bias)
            for r in range(1, R + 1)
        ]
    )
    xr = torch.tensor([2 ** (r + 1 - data_format.bias) for r in range(1, R // 2 + 2)])
    xr[-1] = G[-1]
    xr = torch.concat((-torch.flip(xr, [0]), xr))
    return xr, vr


def snr_uniform(C, sigma2, N):
    C = torch.tensor(C)

    z = C**2 / sigma2
    return 1 / (
        2 * (1 + z) * q_function(torch.sqrt(z))
        - torch.sqrt(2 * z / np.pi) * torch.exp(-0.5 * z)
        + z / (3 * ((N - 1) ** 2))
    )


@cache
def get_copt_uniform(data_format: "UniformDataFormat") -> float:
    C = np.linspace(1, 100, 10000)
    gres = snr_uniform(C, 1, data_format.n_values)
    return C[np.argmax(gres)]


@cache
def get_copt_float(data_format: "FloatDataFormat") -> float:

    C = np.linspace(1, 100, 10000)
    xr, vr = compute_float_grid(data_format)
    gres = snr_float(data_format.get_representable_values(), xr, vr, C, 1.0)

    return C[np.argmax(gres)]
