from dataclasses import dataclass
from typing import Optional

from quant_mp.algs import ALGORITHMS
from quant_mp.algs.template import Algorithm
from quant_mp.datatypes.template import DataFormat


@dataclass(frozen=True)
class QuantConfig:
    qval_data_format: DataFormat
    qparam_data_format: DataFormat
    algorithm: Optional[Algorithm] = None
    symmetric: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":
        qval_data_format = get_data_format(data["qval_data_format"])
        pass

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
