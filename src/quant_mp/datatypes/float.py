from functools import cache

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
        # TODO: Implement logic to compute representable values for float data format
        pass


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
