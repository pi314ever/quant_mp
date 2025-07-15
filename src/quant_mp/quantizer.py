import itertools
from typing import List, Optional

import torch

from quant_mp.datatypes.template import DataFormat


def get_int_values(total_bits: int) -> List[int]:
    n_values = 2**total_bits
    return list(range(-n_values // 2, n_values // 2))


def get_fp_values(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    """Adapted from https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a06a0f6a08cb23754b110359a109e069fa97ce9e/bitsandbytes/functional.py#L258"""
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
    pvalues = []
    for i, val in enumerate(
        range(-(2 ** (exponent_bits - has_sign)), 2 ** (exponent_bits - has_sign), 1)
    ):
        evalues.append(2**val)

    values = []
    lst = list(itertools.product([0, 1], repeat=precision_bits))
    # for ev in evalues:
    bias = 2 ** (exponent_bits - 1)
    for evalue in range(2 ** (exponent_bits)):
        for bit_pattern in lst:
            value = 1 if evalue != 0 else 0
            for i, pval in enumerate(list(bit_pattern)):
                value += pval * (2 ** -(i + 1))
            if evalue == 0:
                # subnormals
                value = value * 2**-(bias)
            else:
                # normals
                value = value * 2 ** -(evalue - bias - 1)
            values.append(value)
            if signed:
                values.append(-value)

    assert len(values) == 2**total_bits
    values.sort()
    return values


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
