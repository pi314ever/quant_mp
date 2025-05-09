from quant_mp.config import rconfig, qconfig

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
    rconfig(
        label="FP32",
        activation=qconfig(qtype=None),
        weight=qconfig(qtype=None),
        grad=qconfig(qtype=None),
    ),
    rconfig(
        label="FP4-minmax",
        activation=qconfig(qtype=qtype, alg="minmax", format=format),
        weight=qconfig(qtype=qtype, alg="minmax", format=format),
        grad=qconfig(qtype=None),
    ),
    rconfig(
        label="FP4-analytic",
        activation=qconfig(qtype=qtype, alg="iterative", format=format),
        weight=qconfig(qtype=qtype, alg="normal", format=format),
        grad=qconfig(qtype=None),
    ),
    rconfig(
        label="FP4-iterative",
        activation=qconfig(qtype=qtype, alg="iterative", format=format),
        weight=qconfig(qtype=qtype, alg="iterative", format=format),
        grad=qconfig(qtype=None),
    ),
]
