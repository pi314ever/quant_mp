from quant_mp.algs import get_algorithm
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes import fp4_e2m1, fp32, int4

model_name = "ResNet"
qblock_size = None
dformat = int4

save_name = (
    "exps/results/"
    + "qat_"
    + dformat.name
    + "_"
    + str(qblock_size)
    + "_"
    + model_name
    + ".pickle"
)

qconfigs = [
    QuantModuleConfig(
        activation=None,
        weight=None,
    ),
    QuantModuleConfig(
        activation=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("minmax"),
            qblock_size=qblock_size,
        ),
        weight=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("minmax"),
            qblock_size=qblock_size,
        ),
    ),
    QuantModuleConfig(
        activation=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("iterative"),
            qblock_size=qblock_size,
        ),
        weight=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("analytic"),
            qblock_size=qblock_size,
        ),
    ),
    QuantModuleConfig(
        activation=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("iterative"),
            qblock_size=qblock_size,
        ),
        weight=QuantConfig(
            qval_data_format=dformat,
            qparam_data_format=fp32,
            algorithm=get_algorithm("iterative"),
            qblock_size=qblock_size,
        ),
    ),
]
