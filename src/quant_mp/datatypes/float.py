from functools import cache
import itertools
from typing import List, Optional

import torch

from .template import DataFormat, register_data_format

class FloatDataFormat(DataFormat):
    signed: bool
    exponent: int
    mantissa: int
    correction_factor: float = 0
    inf: bool
    nan: bool

    def __str__(self) -> str:
        return f"fp{self.bit_width}_e{self.exponent}m{self.mantissa}"

    @property
    def max_value(self) -> float:
        return self.get_representable_values()[-1]

    @property
    def min_value(self) -> float:
        return self.get_representable_values()[0]

    @property
    def n_values(self) -> int:
        return len(self.get_representable_values())

    @property
    def bias(self) -> int:
        """
        Returns the bias for the exponent in this floating-point format.
        The bias is used to represent both positive and negative exponents.
        """
        return 2 ** (self.exponent - 1) - 1

    def cast(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.clamp(data, self.min_value, self.max_value)
        log_abs_data = torch.log2(torch.abs(data))
        underflow_mask = torch.floor(log_abs_data + self.bias) < 1
        step_size = 2 ** (torch.floor(log_abs_data) - self.mantissa)
        step_size[underflow_mask] = 2 ** (1 - self.mantissa - self.bias)

        return step_size * torch.round(data / step_size)

    @cache
    def get_representable_values(self) -> list[float]:
        return get_fp_values(self.signed, self.exponent, self.mantissa, self.bit_width)


@register_data_format
class Fp4_e3m0(FloatDataFormat):
    name = "fp4_e3m0"
    bit_width = 4
    exponent = 3
    mantissa = 0
    signed = True
    inf = False
    nan = False


@register_data_format
class Fp4_e2m1(FloatDataFormat):
    name = "fp4_e2m1"
    bit_width = 4
    exponent = 2
    mantissa = 1
    signed = True
    inf = False
    nan = False


@register_data_format
class Fp8_e5m2(FloatDataFormat):
    name = "fp8_e5m2"
    bit_width = 8
    exponent = 5
    mantissa = 2
    correction_factor = 4
    signed = True
    inf = True
    nan = True


@register_data_format
class Fp8_e4m3(FloatDataFormat):
    name = "fp8_e4m3"
    bit_width = 8
    exponent = 4
    mantissa = 3
    correction_factor = 1
    signed = True
    inf = True
    nan = True


def get_fp_values(signed=True, exponent_bits=5, precision_bits=2, total_bits=8):
    """Adapted from https://github.com/bitsandbytes-foundation/bitsandbytes/blob/a06a0f6a08cb23754b110359a109e069fa97ce9e/bitsandbytes/functional.py#L258"""
    e = exponent_bits
    p = precision_bits
    has_sign = 1 if signed else 0
    assert e + p == total_bits - has_sign
    # the exponent is biased to 2^(e-1) -1 == 0
    evalues = []
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
