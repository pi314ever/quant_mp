from functools import cache
from abc import ABC, abstractmethod

import torch


class DataFormat(ABC):
    name: str
    bit_width: int

    def get_torch_equivalent(self) -> torch.dtype | None:
        """
        Returns the torch equivalent datatype, if it exists.
        """
        return None

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    @abstractmethod
    def max_value(self) -> float:
        """
        Returns the maximum representable value for this data format.
        """

    @property
    @abstractmethod
    def min_value(self) -> float:
        """
        Returns the minimum representable value for this data format.
        """

    @property
    def abs_max_value(self) -> float:
        """
        Returns the absolute maximum representable value for this data format.
        This is the maximum of the absolute values of min_value and max_value.
        """
        return max(abs(self.min_value), abs(self.max_value))

    @property
    @abstractmethod
    def n_values(self) -> int:
        """
        Returns the number of representable values for this data format.
        This is used to determine the quantization level.
        """

    @abstractmethod
    def cast(self, data: torch.Tensor) -> torch.Tensor:
        """
        Casts the data to the nearest representable value in this format.
        """

    def get_output_mask(self, pre_cast_data: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask for the output tensor that indicates which values are valid in this data format.
        This is used in STE (Straight-Through Estimator) to determine which values can be quantized without loss of information.
        """
        return (pre_cast_data <= self.max_value) * (pre_cast_data >= self.min_value)

    @abstractmethod
    @cache
    def get_representable_values(self) -> list[float]:
        """
        Returns all representable values for this data format. Implementation should return sorted values and is cached for performance.
        """


DATA_FORMATS: dict[str, DataFormat] = {}


def register_data_format(cls) -> type[DataFormat]:
    """
    Decorator to register a DataFormat class.
    """
    if not issubclass(cls, DataFormat):
        raise TypeError(f"{cls.__name__} must be a subclass of DataFormat")

    DATA_FORMATS[cls.name] = cls()
    return cls


@cache
def get_data_format(name: str) -> DataFormat:
    if name not in DATA_FORMATS:
        raise RuntimeError(f"Unrecognized data format name: {name}")
    return DATA_FORMATS[name]
