import torch
from quant_mp.quantizer import quantizer
import matplotlib.pyplot as plt


x = 3*torch.randn([64,64]).flatten()


qconfig = {'qtype': 'float', 'qbits': 4, 'alg': 'normal', 'qblock_size':None, 'format': 'e3m0'}
quant_obj = quantizer(qconfig=qconfig)
quant_obj.fit_and_quant(x)
C = quant_obj.G[-1]
sigma2 = torch.linspace((C/10)**2,C**2,100000)
res4 = quant_obj.snr(C, sigma2, quant_obj.xr, quant_obj.vr)
zeta4 = sigma2[torch.argmax(res4)]
plt.plot(C / torch.sqrt(sigma2), res4, label='4-bit float (E3M0)')


qconfig = {'qtype': 'float', 'qbits': 4, 'alg': 'normal', 'qblock_size':None, 'format': 'e2m1'}
quant_obj = quantizer(qconfig=qconfig)
quant_obj.fit_and_quant(x)
C = quant_obj.G[-1]
sigma2 = torch.linspace((C/10)**2,C**2,100000)
res4 = quant_obj.snr(C, sigma2, quant_obj.xr, quant_obj.vr)
zeta4 = sigma2[torch.argmax(res4)]
plt.plot(C / torch.sqrt(sigma2), res4, label='4-bit float (E2M1)')


qconfig = {'qtype': 'uniform', 'qbits': 4, 'alg': 'normal', 'qblock_size':None, 'format': 'e3m0'}
quant_obj = quantizer(qconfig=qconfig)

z = torch.linspace(1,100,10000)

res4 = quant_obj.snr(z, int(2**4))
res3 = quant_obj.snr(z, int(2**3))
res2 = quant_obj.snr(z, int(2**2))
res8 = quant_obj.snr(z, int(2**8))

plt.plot(torch.sqrt(z), res4, label='4-bit uniform', linestyle='--')
plt.plot(torch.sqrt(z), res3, label='3-bit uniform', linestyle='--')
plt.plot(torch.sqrt(z), res2, label='2-bit uniform', linestyle='--')

plt.yscale('log')
plt.xlabel('Clipping point')
plt.legend()
plt.grid()
plt.ylabel('SNR')
plt.savefig('exps/results/snr_vs_clip.jpg', bbox_inches='tight')
plt.show()





