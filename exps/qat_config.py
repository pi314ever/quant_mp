from quant_mp.config import QuantLinearConfig, QuantConfig

model_name = "ResNet"
qbits = 4
qtype = "float"
format = "e2m1"
qblock_size = None

save_name = (
    "exps/results/"
    + "qat_"
    + qtype
    + "_"
    + str(qbits)
    + "_"
    + str(qblock_size)
    + ("_" + format if qtype == "float" else "")
    + "_"
    + model_name
    + ".pickle"
)

qconfigs = [
    QuantLinearConfig(
        label="FP32",
        activation=QuantConfig(qtype=None),
        weight=QuantConfig(qtype=None),
        grad=QuantConfig(qtype=None),
    ),
    QuantLinearConfig(
        label="FP4-minmax",
        activation=QuantConfig(qtype=qtype, alg="minmax", format=format),
        weight=QuantConfig(qtype=qtype, alg="minmax", format=format),
        grad=QuantConfig(qtype=None),
    ),
    QuantLinearConfig(
        label="FP4-analytic",
        activation=QuantConfig(qtype=qtype, alg="iterative", format=format),
        weight=QuantConfig(qtype=qtype, alg="normal", format=format),
        grad=QuantConfig(qtype=None),
    ),
    QuantLinearConfig(
        label="FP4-iterative",
        activation=QuantConfig(qtype=qtype, alg="iterative", format=format),
        weight=QuantConfig(qtype=qtype, alg="iterative", format=format),
        grad=QuantConfig(qtype=None),
    ),
]
