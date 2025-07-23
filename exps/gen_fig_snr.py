import matplotlib.pyplot as plt
import torch

from quant_mp.algs.analytic import snr_float, snr_uniform
from quant_mp.datatypes import Fp4_e2m1, Fp4_e3m0, Int2, Int3, Int4

C = torch.linspace(1, 10, 100)

data_format = Fp4_e3m0()
G = data_format.get_representable_values()
xr, vr = data_format.compute_interval_step_size()
res = snr_float(G, xr, vr, C, 1.0)
plt.plot(C, res, label="4-bit float (E3M0)")

data_format = Fp4_e2m1()
G = data_format.get_representable_values()
xr, vr = data_format.compute_interval_step_size()
res = snr_float(G, xr, vr, C, 1.0)
plt.plot(C, res, label="4-bit float (E2M1)")

data_format = Int4()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
plt.plot(C, res, label="4-bit uniform", linestyle="--")

data_format = Int3()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
plt.plot(C, res, label="3-bit uniform", linestyle="--")

data_format = Int2()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
plt.plot(C, res, label="2-bit uniform", linestyle="--")


plt.yscale("log")
plt.xlabel("Clipping point")
plt.legend()
plt.grid()
plt.ylabel("SNR")
plt.savefig("snr_vs_clip.jpg", bbox_inches="tight")
plt.show()
