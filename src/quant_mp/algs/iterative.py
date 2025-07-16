from typing import TYPE_CHECKING, Optional

import torch

from quant_mp.quantizer import quant

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Iterative(Algorithm):
    name = "iterative"
    has_fit_params = True
    has_custom_gradients = True
    num_iters: int

    def __init__(self, num_iters: int = 1) -> None:
        self.num_iters = num_iters
        super().__init__()

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for _ in range(self.num_iters):
            x_quant = quant(data_format, input, scale, shift)
            scale = torch.sum((input - shift) * x_quant, axis=1) / torch.sum(
                x_quant**2, axis=1
            )
            if shift is not None:
                num_z = torch.sum(input - scale * x_quant, axis=1)
                denum_z = len(input)
                shift = num_z / denum_z
        return scale, shift

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
