from .int import Int2, Int3, Int4, Int8, UniformDataFormat
from .float import Fp4_e3m0, Fp4_e2m1, Fp8_e4m3, Fp8_e5m2, FloatDataFormat

int2 = Int2()
int3 = Int3()
int4 = Int4()
int8 = Int8()

fp4_e3m0 = Fp4_e3m0()
fp4_e2m1 = Fp4_e2m1()
fp8_e5m2 = Fp8_e5m2()
fp8_e4m3 = Fp8_e4m3()

__all__ = [
    "UniformDataFormat",
    "FloatDataFormat",
    "int2",
    "int3",
    "int4",
    "int8",
    "fp4_e3m0",
    "fp4_e2m1",
    "fp8_e5m2",
    "fp8_e4m3",
]
