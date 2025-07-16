from functools import cache
from typing import Generator
from loguru import logger

import torch

from .template import DataFormat, register_data_format


class FloatDataFormat(DataFormat):
    signed: bool
    exponent: int
    mantissa: int
    inf: bool
    nan: bool
    zero_bit_pattern: int = 0b0000000
    inf_bit_pattern: int
    nan_bit_patterns: tuple[int, ...]
    correction_factor: float = 0

    def _validate(self) -> None:
        assert self.bit_width == self.exponent + self.mantissa + (
            1 if self.signed else 0
        ), (
            f"Invalid floating point data configuration: {self.name} has bit width {self.bit_width}, but exponent {self.exponent} and mantissa {self.mantissa} do not match."
        )
        assert self.exponent >= 0, (
            f"Expected non-negative exponent, got {self.exponent}."
        )
        assert self.mantissa >= 0, (
            f"Expected non-negative mantissa, got {self.mantissa}."
        )
        if self.inf:
            assert hasattr(self, "inf_bit_pattern"), (
                f"Data format {self.name} has inf=True but no inf_bit_pattern defined."
            )
        if self.nan:
            assert hasattr(self, "nan_bit_patterns"), (
                f"Data format {self.name} has nan=True but no nan_bit_patterns defined."
            )

    def __str__(self) -> str:
        if hasattr(self, "name"):
            return self.name
        return f"fp{self.bit_width}_e{self.exponent}m{self.mantissa}"

    @property
    def max_value(self) -> float:
        return self.get_representable_values()[-1]

    @property
    def min_value(self) -> float:
        return self.get_representable_values()[0]

    @property
    def max_subnormal(self) -> float:
        """
        Returns the maximum subnormal value for this floating-point format.
        Subnormal values are those that are too small to be represented in normal form.
        """
        return 2 ** (-self.mantissa) * 2 ** (-self.bias)

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
        orig_shape = data.shape
        data = torch.clamp(data, self.min_value, self.max_value)
        data_flat = data.view(-1)
        representable_values_tensor = torch.tensor(self.get_representable_values())
        diffs = (data_flat[:, None] - representable_values_tensor[None, :]).abs()
        indices = torch.argmin(diffs, dim=1)
        return representable_values_tensor[indices].view(orig_shape)

    @cache
    def get_representable_values(self) -> list[float]:
        values = []

        nan_found, inf_found = False, False
        for pattern, s, e, m in self.bit_pattern_range():
            logger.debug(
                f"Processing bit pattern: {pattern:0{self.bit_width}b} (s={s:b}, e={e:0{self.exponent}b}, m={m:0{self.mantissa}b})"
            )
            non_signed_pattern = pattern & ((1 << (self.bit_width - 1)) - 1)
            # NaN representation
            if self.nan and non_signed_pattern in self.nan_bit_patterns:
                nan_found = True
                logger.debug(f"NaN found in pattern: {pattern:0{self.bit_width}b}")
                continue

            # Infinity representation
            if self.inf and non_signed_pattern == self.inf_bit_pattern:
                inf_found = True
                logger.debug(f"Inf found in pattern: {pattern:0{self.bit_width}b}")
                # values.append((-1) ** s * float("inf"))
                continue

            # Zero representation
            if non_signed_pattern == self.zero_bit_pattern:
                logger.debug(f"Zero found in pattern: {pattern:0{self.bit_width}b}")
                values.append((-1) ** s * 0.0)
                continue

            # IEEE 754 representation
            implicit_bit = 1 if e != 0 else 0
            value = (
                (-1) ** s
                * (implicit_bit + m / (2**self.mantissa))
                * (2 ** (e - self.bias))
            )
            logger.debug(
                f"Adding standard value: {value} from pattern {pattern:0{self.bit_width}b}"
            )
            values.append(value)

        if self.nan and not nan_found:
            logger.warning(
                f"Data format {self.name} expected NaN but does not have a NaN representation."
            )
        if self.inf and not inf_found:
            logger.warning(
                f"Data format {self.name} expected Inf but does not have an Inf representation."
            )

        values.sort()
        logger.info(f"Generated representable values: {values}")
        return values

    def bit_pattern_range(self) -> Generator[tuple[int, int, int, int], None, None]:
        """
        Yields the complete bit pattern and the individual components (sign, exponent, mantissa)
        """
        for s in [0, 1] if self.signed else [0]:
            for e in range(2**self.exponent):
                for m in range(2**self.mantissa):
                    pattern = (
                        s << (self.exponent + self.mantissa) | e << self.mantissa | m
                    )
                    yield (pattern, s, e, m)


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
    torch_equivalent = torch.float8_e5m2
    exponent = 5
    mantissa = 2
    correction_factor = 4
    signed = True
    inf = True
    nan = True
    inf_bit_pattern = 0b1111100
    nan_bit_patterns = (0b1111101, 0b1111110, 0b1111111)


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
    inf_bit_pattern = 0b1111111
    nan_bit_patterns = (0b1111111,)


@register_data_format
class Fp8_e4m3fnuz(FloatDataFormat):
    name = "fp8_e4m3fnuz"
    bit_width = 8
    torch_equivalent = torch.float8_e4m3fnuz
    exponent = 4
    mantissa = 3
    signed = True
    inf = False
    nan = False

    @property
    def bias(self):
        # Override bias to match the expected values from torch.float8_e4m3fnuz
        return 8
