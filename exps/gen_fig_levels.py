import torch
from quant_mp.quantizer import get_quantizer
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from quant_mp.config import QuantConfig

sigma = 1
mu = 0.5
x = mu + torch.randn((1000,)) * sigma


rng = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(rng, stats.norm.pdf(rng, mu, sigma))
step = 0.02


qconfig_ = QuantConfig(qtype="float", qbits=4, alg="normal", format="e2m1")
quant_obj = get_quantizer(qconfig=qconfig_)
quant_obj.sym = False
quant_obj.fit_and_quant(x)
lk = quant_obj.compute_quant_levels()
plt.scatter(lk[0], 1 * step * torch.ones(lk.shape[1]), label="Float-e2m1")

qconfig_ = QuantConfig(qtype="float", qbits=4, alg="normal", format="e3m0")

quant_obj = get_quantizer(qconfig=qconfig_)
quant_obj.sym = False
quant_obj.fit_and_quant(x)
lk = quant_obj.compute_quant_levels()
plt.scatter(lk[0], 2 * step * torch.ones(lk.shape[1]), label="Float-e3m0")


qconfig_ = QuantConfig(qtype="uniform", qbits=4, alg="normal")
quant_obj = get_quantizer(qconfig=qconfig_)
quant_obj.sym = False
quant_obj.fit_and_quant(x)
lk = quant_obj.compute_quant_levels()
plt.scatter(lk[0], 3 * step * torch.ones(lk.shape[1]), label="Uniform-4")

plt.legend()
plt.yticks([])
plt.savefig("grids_normal.jpg", bbox_inches="tight")
plt.show()
