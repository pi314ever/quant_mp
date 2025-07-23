import math
from typing import TYPE_CHECKING, Optional

import torch

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


# TODO: Implement LSQ algorithm properly
@register_algorithm
class LSQ(Algorithm):
    name = "lsq"
    has_custom_gradients = True
    eps: float = 1e-5

    def __init__(self, eps: Optional[float] = None):
        super().__init__()
        if eps is not None:
            self.eps = eps

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
        Modified from Learned Step-size Quantization.
        https://arxiv.org/abs/1902.08153
        """
        scale = torch.where(scale > self.eps, scale, self.eps)
        grad_scale = 1.0 / math.sqrt(input.numel() * data_format.max_value)
        q_w = input / scale
        indicate_small = (q_w < data_format.min_value).float()
        indicate_big = (q_w > data_format.max_value).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = (
            (
                indicate_small * data_format.min_value
                + indicate_big * data_format.max_value
                + indicate_middle * (-q_w + data_format.cast(q_w))
            )
            * grad_output
            * grad_scale
        )
        grad_alpha = torch.sum(grad_alpha, dim=1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None
