from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist

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
    eps: float

    def __init__(self, num_iters: int = 1, eps: float = 1e-5) -> None:
        self.num_iters = num_iters
        self.eps = eps
        super().__init__()

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        orig_scale_dtype = scale.dtype
        orig_shift_dtype = shift.dtype if shift is not None else None
        for _ in range(self.num_iters):
            x_quant, _ = quant(data_format, input, scale, shift)
            _shift = 0 if shift is None else shift

            sum_x = torch.sum(
                (input - _shift) * x_quant, dim=1, keepdim=True, dtype=torch.float32
            )
            sum_x_quant_sq = torch.sum(x_quant**2, dim=1, keepdim=True, dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_x_quant_sq, op=dist.ReduceOp.SUM)

            denom = sum_x_quant_sq + self.eps
            scale = (sum_x / denom).to(orig_scale_dtype)
            if shift is not None:
                # TODO: Add distributed version of this
                num_z = torch.sum(
                    (input - scale * x_quant).float(), dim=1, keepdim=True
                )
                denum_z = len(input)
                shift = (num_z / max(denum_z, 1)).to(orig_shift_dtype)
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
