import itertools
from typing import List, Optional

import torch

from quant_mp.datatypes.template import DataFormat


def quant(
    data_format: DataFormat,
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input from the global data format to the quantized data format. Scale and shift (if exists) must be broadcastable to the input.

    Returns a tuple of casted data and cast mask
    """
    out = input.clone()
    if shift is not None:
        out -= shift
    out /= scale
    return data_format.cast(out), data_format.get_output_mask(out)


def dequant(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Dequantizes the input tensor from the quantized data format to the global data format. Scale and shift (if exists) must be broadcastable to the input.
    """
    out = scale * input
    if shift is not None:
        out += shift
    return out
