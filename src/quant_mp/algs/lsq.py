from typing import TYPE_CHECKING

import torch

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


# TODO: Implement LSQ algorithm properly
@register_algorithm
class LSQ(Algorithm):
    name = "lsq"
    has_custom_gradients = True

    def compute_gradients(
        self,
        ctx,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        grad_output: tuple[torch.Tensor],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        pass
