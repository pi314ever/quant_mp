import torch
from quant_mp.quantizer import quantizer
import matplotlib.pyplot as plt
from quant_mp.config import qconfig

C = torch.linspace(1,10,100)

qconfig_ = qconfig(qtype='float', qbits= 4, alg= 'normal', format= 'e3m0')
quant_obj = quantizer(qconfig=qconfig_)
res4 = quant_obj.snr(C, 1.)
plt.plot(C, res4, label='4-bit float (E3M0)')


qconfig_ = qconfig(qtype='float', qbits= 4, alg= 'normal', format= 'e2m1')
quant_obj = quantizer(qconfig=qconfig_)
res4 = quant_obj.snr(C, 1.)
plt.plot(C, res4, label='4-bit float (E2M1)')


qconfig_ = qconfig(qtype='uniform', qbits= 4, alg= 'normal')
quant_obj = quantizer(qconfig=qconfig_)

res4 = quant_obj.snr(C, 1., int(2**4))
res3 = quant_obj.snr(C, 1., int(2**3))
res2 = quant_obj.snr(C, 1., int(2**2))
res8 = quant_obj.snr(C, 1., int(2**8))

plt.plot(C, res4, label='4-bit uniform', linestyle='--')
plt.plot(C, res3, label='3-bit uniform', linestyle='--')
plt.plot(C, res2, label='2-bit uniform', linestyle='--')

plt.yscale('log')
plt.xlabel('Clipping point')
plt.legend()
plt.grid()
plt.ylabel('SNR')
plt.savefig('snr_vs_clip.jpg', bbox_inches='tight')
plt.show()





