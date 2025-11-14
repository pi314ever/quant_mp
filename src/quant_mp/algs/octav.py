from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Octav(Algorithm):
    name = "octav"
    has_fit_params = True
    has_custom_gradients = True
    num_iters: int

    def __init__(self, num_iters: int = 10) -> None:
        self.num_iters = num_iters
        super().__init__()

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        dtype = scale.dtype
        shape = scale.shape
        for _ in range(self.num_iters):
            outside_mask = torch.abs(input) > (scale * data_format.max_value)
            inside_mask = ~outside_mask
            sum_abs_input = torch.sum(
                torch.abs(input) * outside_mask, dim=1, keepdim=True, dtype=dtype
            )
            count_inside = torch.sum(inside_mask, dim=1, keepdim=True, dtype=dtype)
            count_outside = torch.sum(outside_mask, dim=1, keepdim=True, dtype=dtype)
            if dist.is_initialized():
                dist.all_reduce(sum_abs_input, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_inside, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_outside, op=dist.ReduceOp.SUM)
            if shift is None:
                scale = sum_abs_input / (
                    4**-data_format.bit_width / 3 * count_inside + count_outside + 1e-8
                )
            else:
                raise NotImplementedError("Non-symmetric is not implemented for octav")
        return scale.reshape(shape), shift

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
        grad_input, _, _ = self.ste(ctx, quant_mask, grad_output)
        if grad_input is not None:
            outside_mask = ~quant_mask
            grad_input += scale / torch.abs(input + 1e-8) * outside_mask * grad_output
        return grad_input, None, None
