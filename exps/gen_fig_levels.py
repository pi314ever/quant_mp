import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from quant_mp.algs.analytic import Analytic
from quant_mp.datatypes import Fp4_e2m1, Fp4_e3m0, Int4

sigma = 1
mu = 0.5
x = mu + torch.randn((1, 1000)) * sigma

rng = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
plt.plot(rng, stats.norm.pdf(rng, mu, sigma))
step = 0.02

alg = Analytic()
data_format = Fp4_e2m1()
s, z = alg.fit_params(data_format, x, torch.tensor(1.0), torch.tensor(0.0))
lk = s * data_format.get_representable_values() + z
plt.scatter(lk, 1 * step * torch.ones(lk.shape[0]), label="Float-e2m1")

alg = Analytic()
data_format = Fp4_e3m0()
s, z = alg.fit_params(data_format, x, torch.tensor(1.0), torch.tensor(0.0))
lk = s * data_format.get_representable_values() + z
plt.scatter(lk, 2 * step * torch.ones(lk.shape[0]), label="Float-e3m0")


alg = Analytic()
data_format = Int4()
s, z = alg.fit_params(data_format, x, torch.tensor(1.0), torch.tensor(0.0))
lk = s * data_format.get_representable_values() + z
plt.scatter(lk, 3 * step * torch.ones(lk.shape[0]), label="Int4")

plt.legend()
plt.yticks([])
plt.savefig("grids_normal.jpg", bbox_inches="tight")
plt.show()
