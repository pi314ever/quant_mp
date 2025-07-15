from typing import TYPE_CHECKING, Optional

import torch

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class MinMax(Algorithm):
    name = "minmax"
    has_update_params = True
    has_custom_gradients = True

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,  # pyright: ignore[reportDeprecated]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: Validate if axis=1 is a valid assumption. If not, how do we generalize this properly?
        max_x = torch.max(torch.abs(input), axis=1)[0]
        if shift is None:
            return 2 * max_x / (data_format.max_value - data_format.min_value), None
        min_x = torch.min(input, axis=1)[0]
        # TODO: Check if this shift calculation is correct
        return (max_x - min_x) / (
            data_format.max_value - data_format.min_value
        ), data_format.min_value - min_x / scale

        # NOTE: Old implementation
        from quant_mp.datatypes import FloatDataFormat, UniformDataFormat

        if not isinstance(data_format, (UniformDataFormat, FloatDataFormat)):
            raise RuntimeError(f"Invalid data format: {data_format}")

        max_x = torch.max(torch.abs(input), axis=1)[0]
        if isinstance(data_format, UniformDataFormat):
            if shift is None:
                return 2 * max_x / (data_format.n_values - 1), None
            # Non-symmetric
            min_x = torch.min(input, axis=1)[0]
            return (max_x - min_x) / (data_format.n_values - 1), min_x + scale / 2
        # Float
        if shift is None:
            return max_x / data_format.max_value
        min_x = torch.min(input, axis=1)[0]
        return (max_x - min_x) / (
            2 * data_format.max_value
        ), min_x + data_format.max_value * scale

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
