import matplotlib.pyplot as plt
import torch

from quant_mp.config import QuantConfig
from quant_mp.quantizer import get_quantizer

# FIXME: Update to new API

C = torch.linspace(1, 10, 100)

qconfig_ = QuantConfig(qtype="float", qbits=4, algorithm="normal", format="e3m0")
quant_obj = get_quantizer(qconfig=qconfig_)
res4 = quant_obj.snr(C, 1.0)
plt.plot(C, res4, label="4-bit float (E3M0)")


qconfig_ = QuantConfig(qtype="float", qbits=4, algorithm="normal", format="e2m1")
quant_obj = get_quantizer(qconfig=qconfig_)
res4 = quant_obj.snr(C, 1.0)
plt.plot(C, res4, label="4-bit float (E2M1)")


qconfig_ = QuantConfig(qtype="uniform", qbits=4, algorithm="normal")
quant_obj = get_quantizer(qconfig=qconfig_)

res4 = quant_obj.snr(C, 1.0, int(2**4))
res3 = quant_obj.snr(C, 1.0, int(2**3))
res2 = quant_obj.snr(C, 1.0, int(2**2))
res8 = quant_obj.snr(C, 1.0, int(2**8))

plt.plot(C, res4, label="4-bit uniform", linestyle="--")
plt.plot(C, res3, label="3-bit uniform", linestyle="--")
plt.plot(C, res2, label="2-bit uniform", linestyle="--")

plt.yscale("log")
plt.xlabel("Clipping point")
plt.legend()
plt.grid()
plt.ylabel("SNR")
plt.savefig("snr_vs_clip.jpg", bbox_inches="tight")
plt.show()
