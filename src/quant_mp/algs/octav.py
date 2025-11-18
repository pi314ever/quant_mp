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
    _accumulation_dtype = torch.float32
    _eps = 1e-8

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
        device = scale.device
        work_dtype = (
            self._accumulation_dtype
            if torch.finfo(dtype).bits < torch.finfo(self._accumulation_dtype).bits
            else dtype
        )
        scale_work = scale.reshape(shape).to(device=device, dtype=work_dtype)
        finfo = torch.finfo(work_dtype)
        df_max = torch.tensor(
            float(data_format.max_value), dtype=work_dtype, device=device
        )
        for _ in range(self.num_iters):
            input_work = input.to(device=device, dtype=work_dtype)
            outside_mask = torch.abs(input_work) > (scale_work * df_max)
            inside_mask = ~outside_mask
            sum_abs_input = torch.sum(
                torch.abs(input_work) * outside_mask,
                dim=1,
                keepdim=True,
                dtype=work_dtype,
            )
            count_inside = torch.sum(
                inside_mask, dim=1, keepdim=True, dtype=torch.int32
            )
            count_outside = torch.sum(
                outside_mask, dim=1, keepdim=True, dtype=torch.int32
            )
            if dist.is_initialized():
                dist.all_reduce(sum_abs_input, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_inside, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_outside, op=dist.ReduceOp.SUM)
            if shift is None:
                denom = (
                    (4 ** (-data_format.bit_width) / 3.0) * count_inside
                    + count_outside
                    + self._eps
                )
                scale_work = sum_abs_input / denom
                scale_work /= df_max
                scale_work = torch.clamp(scale_work, min=0.0, max=finfo.max)
            else:
                raise NotImplementedError("Non-symmetric is not implemented for octav")
        return scale_work.reshape(shape).to(dtype=dtype), shift

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
            safe_scale = torch.nan_to_num(
                scale.to(dtype=grad_input.dtype),
                nan=0.0,
                posinf=torch.finfo(grad_input.dtype).max,
                neginf=0.0,
            )
            denom = torch.abs(input).to(dtype=grad_input.dtype)
            denom = torch.clamp(denom, min=self._eps)
            correction = safe_scale / denom
            correction = torch.where(
                outside_mask,
                correction,
                torch.zeros_like(correction),
            )
            grad_input = grad_input + correction * grad_output
        return grad_input, None, None
