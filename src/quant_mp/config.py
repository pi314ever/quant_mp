from dataclasses import dataclass
from typing import Optional

from quant_mp.algs import ALGORITHMS
from quant_mp.algs.template import Algorithm, get_algorithm
from quant_mp.datatypes.template import DataFormat, get_data_format


@dataclass(frozen=True)
class QuantConfig:
    qval_data_format: DataFormat  # Primary data format for quantizing values.
    qparam_data_format: DataFormat  # Not currently in use. Will implement later
    algorithm: Optional[Algorithm] = None
    symmetric: bool = True
    qblock_size: None | int | str = None

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":
        qval_data_format = get_data_format(data.pop("qval_data_format"))
        qparam_data_format = get_data_format(data.pop("qparam_data_format"))
        algorithm_init_kwargs = data.pop("algorithm_init_kwargs", None)
        algorithm_name = data.pop("algorithm", QuantConfig.algorithm)
        if algorithm_name is not None:
            algorithm = get_algorithm(
                algorithm_name, algorithm_init_kwargs=algorithm_init_kwargs
            )
        else:
            algorithm = None

        return cls(
            qval_data_format,
            qparam_data_format,
            algorithm,
            **data,
        )

    def __post_init__(self):
        assert self.algorithm in ALGORITHMS, (
            f"Invalid algorithm {self.algorithm}. Valid choices: {ALGORITHMS.keys()}"
        )


@dataclass
class QuantLinearConfig:
    label: str
    activation: Optional[QuantConfig]
    weight: Optional[QuantConfig]

    @classmethod
    def from_dict(cls, data: dict) -> "QuantLinearConfig":
        pass
