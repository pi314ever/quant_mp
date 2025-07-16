from functools import cache
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import torch

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


class Algorithm:
    name: str = "noop"
    has_custom_gradients = False
    has_update_params = False

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,  # pyright: ignore[reportDeprecated]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns updated parameters based on current input, scale, shift, and data format
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement update_params"
        )

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
        Computes a gradient for the algorithm. This will occur within a torch.autocast.Function.backward context.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement compute_gradients"
        )

    def ste(
        self,
        ctx,
        quant_mask: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        """
        Straight Through Estimator (STE) for a standard algorithm.
        """
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * quant_mask

        return grad_input, None, None


ALGORITHMS: dict[str, type[Algorithm]] = {}
_T = TypeVar("_T", bound=type)


def register_algorithm(cls: _T) -> _T:
    """
    Register a new quantization algorithm class.
    """
    if not issubclass(cls, Algorithm):
        raise TypeError(f"{cls.__name__} is not a subclass of Algorithm")
    assert hasattr(cls, "name"), f"{cls.__name__} must have a 'name' attribute"
    if cls.name not in ALGORITHMS:
        ALGORITHMS[cls.name] = cls

    return cls


@cache
def get_algorithm(
    name: str, *, algorithm_init_kwargs: Optional[dict[str, Any]] = None
) -> Algorithm:
    if name not in ALGORITHMS:
        raise RuntimeError(f"Unrecognized algorithm name: {name}")
    alg_kwargs = algorithm_init_kwargs or {}
    return ALGORITHMS[name](**alg_kwargs)
