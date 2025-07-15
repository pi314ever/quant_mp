from dataclasses import dataclass
from typing import Optional

from quant_mp.algs import ALGORITHMS
from quant_mp.algs.template import Algorithm, get_algorithm
from quant_mp.datatypes.template import DataFormat, get_data_format


@dataclass(frozen=True)
class QuantConfig:
    qval_data_format: DataFormat
    qparam_data_format: DataFormat
    algorithm: Optional[Algorithm] = None
    symmetric: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":
        qval_data_format = get_data_format(data["qval_data_format"])
        qparam_data_format = get_data_format(data["qparam_data_format"])
        algorithm_init_kwargs = data.get("algorithm_init_kwargs", None)
        algorithm_name = data.get("algorithm", QuantConfig.algorithm)
        if algorithm_name is not None:
            algorithm = get_algorithm(
                algorithm_name, algorithm_init_kwargs=algorithm_init_kwargs
            )
        else:
            algorithm = None
        symmetric = data.get("symmetric", QuantConfig.symmetric)

        return cls(qval_data_format, qparam_data_format, algorithm, symmetric)

    def __post_init__(self):
        assert self.algorithm in ALGORITHMS, (
            f"Invalid algorithm {self.algorithm}. Valid choices: {ALGORITHMS.keys()}"
        )

    @property
    def is_quantized(self) -> bool:
        return self.qtype is not None

    @property
    def alg_requires_grad_params(self) -> bool:
        return self.algorithm == "lsq"


@dataclass
class QuantLinearConfig:
    label: str
    activation: Optional[QuantConfig]
    weight: Optional[QuantConfig]

    @classmethod
    def from_dict(cls, data: dict) -> "QuantLinearConfig":
        pass
