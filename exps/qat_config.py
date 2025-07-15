from quant_mp.config import QuantConfig, QuantLinearConfig

# FIXME: Update to new architecture

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
        activation=None,
        weight=None,
    ),
    QuantLinearConfig(
        label="FP4-minmax",
        activation=QuantConfig(qtype=qtype, algorithm="minmax", format=format),
        weight=QuantConfig(qtype=qtype, algorithm="minmax", format=format),
    ),
    QuantLinearConfig(
        label="FP4-analytic",
        activation=QuantConfig(qtype=qtype, algorithm="iterative", format=format),
        weight=QuantConfig(qtype=qtype, algorithm="normal", format=format),
    ),
    QuantLinearConfig(
        label="FP4-iterative",
        activation=QuantConfig(qtype=qtype, algorithm="iterative", format=format),
        weight=QuantConfig(qtype=qtype, algorithm="iterative", format=format),
    ),
]
