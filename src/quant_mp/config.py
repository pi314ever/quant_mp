from dataclasses import dataclass
from typing import Optional


@dataclass
class qconfig:
    qtype: Optional[str] = None
    qbits: int = 4
    qblock_size: Optional[int] = None
    alg: str = "minmax"
    beta: float = 0.0
    format: Optional[str] = "e2m1"


@dataclass
class rconfig:
    label: str
    activation: qconfig
    weight: qconfig
    grad: qconfig
