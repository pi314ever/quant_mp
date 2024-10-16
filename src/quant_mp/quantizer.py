
import torch
torch.set_printoptions(precision=5)
import time
from scipy.spatial import distance
import numpy as np
from sklearn.cluster import KMeans
from scipy import special
from abc import ABC, abstractmethod

def quantizer(qconfig):

    quantizers = {
        "uniform": quantizer_uniform,
        "nonuniform": quantizer_nonuniform,
        "float": quantizer_float,
    }
    return quantizers[qconfig['qtype']](qconfig)


class quantizer_base(ABC):
    def __init__(self, qconfig):
        
        self.qconfig = qconfig
        self.b = qconfig['qbits']
        self.alg = qconfig['alg']

        self.N = int(2**self.b)
        self.first_batch = True
        self.params = None

    def error(self, x, xdeq):
        err = torch.sum(((x - xdeq)**2)) / len(x)
        return err
    
    def compute_block_size(self, x):

        self.block_size=None
        if self.qconfig['qblock_size'] is None:
            self.block_size = x.numel()
        elif isinstance(self.qconfig['qblock_size'], int):
            self.block_size = self.qconfig['qblock_size']
        elif self.qconfig['qblock_size'] == 'channel':
            self.block_size = x.shape[-1]
    
    def q_function(self, x):
        return 0.5 - 0.5*special.erf(x/np.sqrt(2))
    
    def gauss_cdf(self, x, m, std):
        return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.)) * std)))
    
    @abstractmethod
    def compute_quant_levels(self):
        pass

    @abstractmethod
    def quant(self):
        pass

    @abstractmethod
    def dequant(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    def fit_and_quant(self, x, params=None):

        self.compute_block_size(x)

        shape_org = x.shape
        x = x.view(-1, self.block_size)

        self.fit(x)
        
        if params is None:
            x = self.quant(x, self.params)
        else:
            x = self.quant(x, params)

        params = self.params

        return x.to(torch.float32).view(shape_org), params
    


class quantizer_uniform(quantizer_base):
    def __init__(self, qconfig):
        super().__init__(qconfig)

        self.fit_dispatcher = {
            "minmax": self.fit_minmax,
            "normal": self.fit_normal,
            "iterative": self.fit_iterative
        }

        self.sym = True
        self.s = None
        self.z = None

        self.k_list = torch.arange(0, self.N-1).to(torch.int)

    def compute_quant_levels(self):
        lk = self.s * self.k_list + self.z
        return lk
    
    def quant(self, x, params):
        s, z = params
        return torch.clamp(torch.round((x - z) / s), -self.N/2+1, self.N/2-1).to(torch.int)

    def dequant(self, xint, params):
        s, z = params
        return s * xint + z
    
    def snr(self, z, N):
        return 1/(2 * (1 + z) * self.q_function(np.sqrt(z)) - np.sqrt(2*z/np.pi) * np.exp(-0.5*z) + z/(3*((N-1)**2)))
    
    def fit(self, x):
        self.fit_dispatcher[self.alg](x)
    
    def fit_minmax(self, x):

        if self.sym:
            self.maxx = torch.max(torch.abs(x), axis=1, keepdim=True)[0]
            self.s = 2 * self.maxx / (self.N-1)
            self.z = 0

        else:
            self.minx = torch.min(x, axis=1, keepdim=True)[0]
            self.maxx = torch.max(x, axis=1, keepdim=True)[0]

            self.s = (self.maxx - self.minx) / (self.N-1)
            self.z = self.minx + self.s/2

        self.params = (self.s, self.z)

    def fit_normal(self, x):

        if not hasattr(self, 'zeta'):
            z = np.linspace(1,100,10000)
            gres = self.snr(z, self.N)
            self.zeta = z[np.argmax(gres)]

        xmean = torch.mean(x, axis=1, keepdim=True)
        xvar = torch.var(x, axis=1, keepdim=True)


        self.s = (2 * torch.sqrt(self.zeta * xvar)) / (self.N-1)
        if self.sym:
            self.z = 0.
        else:
            self.z = xmean

        self.params = (self.s, self.z)
    
    def fit_iterative(self, x):
        
        if self.params is None:
            self.fit_normal(x)

        denum_z = len(x)

        s,z = self.params
        #sz_prev = torch.zeros(2)
        for _ in range(1):
            xint = self.quant(x, (s, z))

            num_s = torch.sum((x - z) * xint, axis=1, keepdim=True)
            denum_s = torch.sum(xint**2, axis=1, keepdim=True)
            num_z = torch.sum(x - s * xint, axis=1, keepdim=True)

            s = num_s / denum_s
            if not self.sym:
                z = num_z / denum_z

            self.s = s
            self.z = z
                
        self.params = (self.s, self.z)


class quantizer_nonuniform(quantizer_base):
    def __init__(self, qconfig):
        super().__init__(qconfig)

        self.fit_dispatcher = {
            "quantile": self.fit_quantile,
            "analytic": self.fit_analytic,
            "iterative": self.fit_iterative
        }

        self.sk_kmeans = True

    def compute_quant_levels(self):
        return self.lk

    def quant(self, x, lk):
        return torch.argmin((x[:,None] - lk)**2, axis=1).to(torch.int)
   
    def dequant(self, xint, lk):
        return lk[torch.arange(len(xint)), xint]
    
    def fit(self, x):
        self.block_size=x.numel()
        self.fit_dispatcher[self.alg](x)

    def fit_quantile(self, x):

        nblocks = len(x) // self.block_size

        x_sorted = torch.sort(x.reshape(-1, self.block_size), axis=1)[0]
        k = torch.arange(0, self.N)
        ind = ((self.block_size-1) * k.to(torch.float64) / (self.N-1)).to(torch.int)
        self.tk = x_sorted[:, ind]
        self.tk[:,0] = torch.nan_to_num(torch.tensor(-float('inf')))
        self.tk[:,-1] = torch.nan_to_num(torch.tensor(float('inf')))
        self.lk = torch.zeros(nblocks, self.N-1)
        for i in k-1:
            self.lk[:,i] = torch.mean(x_sorted[:,ind[i]:ind[i+1]], axis=1)

    def fit_analytic(self, x):

        if not hasattr(self, 'lkopt'):
            tk = torch.quantile(torch.randn(1000), torch.arange(self.N)/(self.N-1))
            tk[0] = torch.nan_to_num(torch.tensor(-float('inf')))
            tk[-1] = torch.nan_to_num(torch.tensor(float('inf')))
            pdf = torch.distributions.normal.Normal(0., 1.)
            tk = tk.to(torch.float64)

            for _ in range(500):

                F = self.gauss_cdf(tk, 0., 1.)
                P = torch.exp(pdf.log_prob(torch.Tensor(tk)))
                
                lk = -(P[1:] - P[:-1]) / (F[1:] - F[:-1])
                tk[1:-1] = (lk[1:] + lk[0:-1])/2

            self.lkopt = lk.to(torch.float32)
            self.tkopt = tk.to(torch.float32)

        xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
        xstd = torch.std(x.reshape(-1, self.block_size), axis=1)

        self.lk = (xstd[:,None] * self.lkopt + xmean[:,None])
        #self.tk = x.mean()

    def fit_iterative(self, x, verbose=False):

        self.fit_nonuniform_analytic(x)
        #self.fit_uniform_minmax(x)

        if not hasattr(self, 'f'):
            self.f = torch.ones(x.shape)

        nblocks = len(x) // self.block_size

        for i in range(nblocks):

            xb = x[i*self.block_size:(i+1)*self.block_size]
            lk = self.lk[i]

            kmeans = KMeans(n_clusters=self.N-1, init=lk.reshape(-1,1), max_iter=500, tol=1e-5)
            kmeans.fit(xb.reshape(-1,1), self.f, )
            self.lk[i] = torch.tensor(kmeans.cluster_centers_[:,0])



class quantizer_float(quantizer_base):
    def __init__(self, qconfig):
        super().__init__(qconfig)

        self.fit_dispatcher = {
            "cast": self.fit_cast,
            "minmax": self.fit_minmax,
            "normal": self.fit_normal,
            "iterative": self.fit_iterative,
        }

        self.format = qconfig['format']
        self.sym = True
        self.s = None
        self.z = None

        dict_format = {
            (8,'e4m3'): (4, 3, 7, 0),
            (8,'e5m2'): (5, 2, 15, 3),
            (4,'e2m1'): (2, 1, 2, 0),
            (4,'e3m0'): (3, 0, 3, 0),
            (16,'fp'):  (5, 10, 15, 0),
            (16,'bf'):  (8, 7, 127, 0),
        }
        self.E, self.M, self.bias, self.c = dict_format[(self.b, self.format)]

        self.float_grid(self.E, self.M, self.bias, self.c) # Need for this?

    def compute_quant_levels(self):
        lk = self.G * self.s + self.z
        return lk

    def quant(self, x, params):
        s, z = params
        return self.cast_to_fp((x - z)/s)

    def dequant(self, xfloat, params):
        s, z = params
        return s * xfloat + z
    
    def cast_to_fp(self, x):

        x = torch.clamp(x, self.G[0].to(x.device), self.G[-1].to(x.device))

        v = 2**(torch.floor(torch.log2(torch.abs(x)) + 2**(-self.bias)) - self.M)
        v[torch.floor(torch.log2(torch.abs(x)) + self.bias) < 1] = 2**(1-self.M-self.bias)
        
        Xf = v * torch.round(x / v)
        
        return Xf
    
    def snr(self, C, sigma2, xr, vr):
        res = 2 * (1 + C**2 / sigma2) * self.q_function(C / torch.sqrt(sigma2)) 
        res += - C * torch.sqrt(torch.tensor(2.)/torch.pi) * torch.exp(-0.5*(C**2)/sigma2) / torch.sqrt(sigma2)

        F = self.gauss_cdf(xr[None], 0., torch.sqrt(sigma2[:,None]))
        p = F[:,1:] - F[:,:-1]

        res += torch.sum(vr**2 * p / (12*sigma2[:,None]), axis=1)
        return 1/res
    
    def float_grid(self, E=8, M=10, bias=15, special=0):

        Gn = [2**(k // 2**M) * 2**(-bias) * (1 + k % (2**M) * 2**(-M)) for k in range(2**M, 2**(M+E)-1-special)]
        Gs = [2**(-bias) * (k * 2**(1-M)) for k in range(1, 2**M)]
        self.Gh = torch.tensor(Gs + Gn)
        self.G = torch.concat((-torch.flip(self.Gh, [0]), torch.tensor([0.]), self.Gh))

    def fit(self, x):
        self.fit_dispatcher[self.alg](x)

    def fit_cast(self, x):

        self.s = torch.tensor(1.)
        self.z = torch.tensor(0.)
        #self.lk = self.G * self.s + self.z
        self.params = (self.s, self.z)

    def fit_minmax(self, x):
        
        if self.sym:
            self.maxx = torch.max(torch.abs(x), axis=1, keepdim=True)[0]
            self.s = 2 * self.maxx / (2*torch.max(self.G))
            self.z = 0

        else:
            self.minx = torch.min(x, axis=1, keepdim=True)[0]
            self.maxx = torch.max(x, axis=1, keepdim=True)[0]

            self.s = (self.maxx - self.minx) / (2*torch.max(self.G))
            self.z = self.minx + torch.max(self.G) * self.s

        self.params = (self.s, self.z)

    def fit_normal(self, x):

        if not hasattr(self, 'sigma2opt'):
            C = self.G[-1]
            kmax = (2**(self.E + self.M) - 2 - self.c)
            self.R = kmax // 2**self.M + (kmax % 2**self.M > 0) * 1 - 1
            self.R = 2 * self.R - 1

            self.vr = torch.tensor([2 ** (abs(r -1 - self.R//2) + 1 - self.M - self.bias) for r in range(1, self.R+1)])

            #self.vr = torch.tensor([2 ** (r - self.M - self.bias) for r in range(1, self.R+1)])
            #self.vr = torch.concat((torch.flip(self.vr[1:], [0]), self.vr))

            #torch.tensor([2**(self.R//2 - r + 3 - self.bias) for r in range(1,self.R//2+2)])
            #torch.tensor([2**(r - self.R//2  - self.bias) for r in range(self.R//2+2, self.R+2)])
            
            self.xr = torch.tensor([2**(r + 1 - self.bias) for r in range(1,self.R//2+2)])
            self.xr[-1] = C
            self.xr = torch.concat((-torch.flip(self.xr, [0]), self.xr))
            sigma2 = torch.linspace(0.1,100,100000)
            gres = self.snr(C, sigma2, self.xr, self.vr)
            self.sigma2opt = sigma2[np.argmax(gres)]

        xmean = torch.mean(x, axis=1, keepdim=True)
        xvar = torch.var(x, axis=1, keepdim=True)

        self.s = torch.sqrt(xvar / self.sigma2opt)
        if self.sym:
            self.z = 0.
        else:
            self.z = xmean

        self.params = (self.s, self.z)

    def fit_iterative(self, x):

        if self.params is None:
            self.fit_normal(x)

        denum_z = len(x)

        s, z = self.params
        # sz_prev = torch.zeros(2)
        for _ in range(1):
            xfloat = self.quant(x, (s, z))
            
            num_s = torch.sum((x - z) * xfloat, axis=1, keepdim=True)
            denum_s = torch.sum(xfloat**2, axis=1, keepdim=True)
            num_z = torch.sum(x - s * xfloat, axis=1, keepdim=True)

            s = num_s / denum_s
            if not self.sym:
                z = num_z / denum_z

            self.s = s
            self.z = z

            # if torch.sum(torch.abs(sz_prev - sz)) / torch.sum(torch.abs(sz)) < 1e-5:
            #     #print('Loss', err)
            #     return
            # sz_prev = sz

        self.params = (s, z)
            
        # print('NOT CONVERGED !!!')
        # print(sz)




# class quantizer_weight():
#     def __init__(self, b, f=None, qtype='float', format_fp4='e3m0', format_fp8='e4m3', format_fp16='fp'):

#         self.b = b
#         self.N = int(2**b)
#         self.qtype = qtype

#         self.sym = True
#         self.first_batch = True
#         #if self.qtype == 'uniform':
#         self.k_list = torch.arange(0, self.N-1).to(torch.int)

#         if self.qtype == 'nonuniform':
#             self.sk_kmeans = True
            
#         self.block_size = None
#         self.s = None
#         self.z = None

#         if self.qtype == 'float':
#             if self.b == 8:
#                 if format_fp8=='e4m3':
#                     self.E=4; self.M=3; self.bias=7; self.c=0
#                 elif format_fp8=='e5m2':
#                     self.E=5; self.M=2; self.bias=15; self.c=3
#             elif self.b == 4:
#                 if format_fp4=='e2m1':
#                     self.E=2; self.M=1; self.bias=2; self.c=0
#                 elif format_fp4=='e3m0':
#                     self.E=3; self.M=0; self.bias=3; self.c=0
#             elif self.b == 16:
#                 if format_fp16 == 'fp':
#                     self.E=5; self.M=10; self.bias=15; self.c=0
#                 elif format_fp16 == 'bf':
#                     self.E=8; self.M=7; self.bias=127; self.c=0

#             self.float_grid(self.E, self.M, self.bias, self.c)

#     def error(self, x, xdeq):
#         if not hasattr(self, 'f'):
#             self.f = torch.ones(x.shape)
#         #err = torch.sum(((x - self.lk[xint])**2) * self.f) / torch.sum(self.f)
#         err = torch.sum(((x - xdeq)**2)) / len(x)
#         return err
    
#     def compute_quant_levels(self):
#         if self.qtype == 'uniform':
#             lk = self.s * self.k_list + self.z
#         elif self.qtype == 'nonuniform':
#             lk = self.lk
#         elif self.qtype == 'float':
#             lk = self.G * self.s + self.z
#         return lk

#     def q_function(self, x):
#         return 0.5 - 0.5*special.erf(x/np.sqrt(2))

#     def snr_uni(self, z, N):
#         return 1/(2 * (1 + z) * self.q_function(np.sqrt(z)) - np.sqrt(2*z/np.pi) * np.exp(-0.5*z) + z/(3*((N-1)**2)))

#     def snr_float(self, C, sigma2, xr, vr):
#         res = 2 * (1 + C**2 / sigma2) * self.q_function(C / torch.sqrt(sigma2)) 
#         res += - C * torch.sqrt(torch.tensor(2.)/torch.pi) * torch.exp(-0.5*(C**2)/sigma2) / torch.sqrt(sigma2)

#         F = self.gauss_cdf(xr[None], 0., torch.sqrt(sigma2[:,None]))
#         p = F[:,1:] - F[:,:-1]

#         res += torch.sum(vr**2 * p / (12*sigma2[:,None]), axis=1)
#         return 1/res

#     def quant_nonuniform(self, x, lk):
#         return torch.argmin((x[:,None] - lk)**2, axis=1).to(torch.int)
   
#     def dequant_nonuniform(self, xint, lk):
#         return lk[torch.arange(len(xint)), xint]

#     def quant_uniform(self, x, s, z):
#         return torch.clamp(torch.round((x - z) / s), -self.N/2+1, self.N/2-1).to(torch.int)

#     def dequant_uniform(self, xint, s, z):
#         return s * xint + z

#     def quant_float(self, x, s, z):
#         return self.cast_to_fp((x - z)/s) 

#     def dequant_float(self, xfloat, s, z):
#         return s * xfloat + z

#     def cast_to_fp(self, x):

#         x = torch.clamp(x, self.G[0].to(x.device), self.G[-1].to(x.device))

#         v = 2**(torch.floor(torch.log2(torch.abs(x)) + 2**(-self.bias)) - self.M)
#         v[torch.floor(torch.log2(torch.abs(x)) + self.bias) < 1] = 2**(1-self.M-self.bias)
        
#         Xf = v * torch.round(x / v)
        
#         return Xf

#     def float_grid(self, E=8, M=10, bias=15, special=0):

#         Gn = [2**(k // 2**M) * 2**(-bias) * (1 + k % (2**M) * 2**(-M)) for k in range(2**M, 2**(M+E)-1-special)]
#         Gs = [2**(-bias) * (k * 2**(1-M)) for k in range(1, 2**M)]
#         self.Gh = torch.tensor(Gs + Gn)
#         self.G = torch.concat((-torch.flip(self.Gh, [0]), torch.tensor([0.]), self.Gh))

#     def gauss_cdf(self, x, m, std):
#         return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.)) * std)))
    
#     def fit_float_cast(self, x):

#         self.s = torch.tensor(1.)
#         self.z = torch.tensor(0.)
#         #self.lk = self.G * self.s + self.z

#     def fit_float_minmax(self, x):
        
#         if self.sym:
#             self.maxx = torch.max(torch.abs(x).reshape(-1, self.block_size), axis=1)[0]
#             self.s = 2 * self.maxx / (2*torch.max(self.G))
#             self.z = 0

#         else:
#             self.minx = torch.min(x.reshape(-1, self.block_size), axis=1)[0]
#             self.maxx = torch.max(x.reshape(-1, self.block_size), axis=1)[0]

#             self.s = (self.maxx - self.minx) / (2*torch.max(self.G))
#             self.z = self.minx + torch.max(self.G) * self.s

#     def fit_float_normal(self, x):

#         if not hasattr(self, 'sigma2opt'):
#             C = self.G[-1]
#             kmax = (2**(self.E + self.M) - 2 - self.c)
#             self.R = kmax // 2**self.M + (kmax % 2**self.M > 0) * 1 - 1
#             self.R = 2 * self.R - 1

#             self.vr = torch.tensor([2 ** (abs(r -1 - self.R//2) + 1 - self.M - self.bias) for r in range(1, self.R+1)])

#             #self.vr = torch.tensor([2 ** (r - self.M - self.bias) for r in range(1, self.R+1)])
#             #self.vr = torch.concat((torch.flip(self.vr[1:], [0]), self.vr))

#             #torch.tensor([2**(self.R//2 - r + 3 - self.bias) for r in range(1,self.R//2+2)])
#             #torch.tensor([2**(r - self.R//2  - self.bias) for r in range(self.R//2+2, self.R+2)])
            
#             self.xr = torch.tensor([2**(r + 1 - self.bias) for r in range(1,self.R//2+2)])
#             self.xr[-1] = C
#             self.xr = torch.concat((-torch.flip(self.xr, [0]), self.xr))
#             sigma2 = torch.linspace(0.1,100,100000)
#             gres = self.snr_float(C, sigma2, self.xr, self.vr)
#             self.sigma2opt = sigma2[np.argmax(gres)]

#         xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
#         xvar = torch.var(x.reshape(-1, self.block_size), axis=1)

#         self.s = torch.sqrt(xvar / self.sigma2opt)
#         if self.sym:
#             self.z = 0.
#         else:
#             self.z = xmean

#     def fit_float_iterative(self, x):

#         if self.s is None:
#             self.fit_float_normal(x)

#         denum_z = len(x)

#         s,z = self.s, self.z
#         # sz_prev = torch.zeros(2)
#         for _ in range(1):
#             xfloat = self.quant_float(x, s, z)

#             num_s = torch.sum((x - z) * xfloat)
#             denum_s = torch.sum(xfloat**2)
#             num_z = torch.sum(x - s * xfloat)

#             s = num_s / denum_s
#             if not self.sym:
#                 z = num_z / denum_z

#             self.s = s
#             self.z = z

#             # if torch.sum(torch.abs(sz_prev - sz)) / torch.sum(torch.abs(sz)) < 1e-5:
#             #     #print('Loss', err)
#             #     return
#             # sz_prev = sz
            
#         # print('NOT CONVERGED !!!')
#         # print(sz)

#     def fit_uniform_minmax(self, x):

#         if self.sym:

#             self.maxx = torch.max(torch.abs(x).reshape(-1, self.block_size), axis=1)[0]
#             self.s = 2 * self.maxx / (self.N-1)
#             self.z = 0

#         else:
#             self.minx = torch.min(x.reshape(-1, self.block_size), axis=1)[0]
#             self.maxx = torch.max(x.reshape(-1, self.block_size), axis=1)[0]

#             self.s = (self.maxx - self.minx) / (self.N-1)
#             self.z = self.minx + self.s/2

#     def fit_uniform_normal(self, x):

#         if not hasattr(self, 'zeta'):
#             z = np.linspace(1,100,10000)
#             gres = self.snr_uni(z, self.N)
#             self.zeta = z[np.argmax(gres)]

#         xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
#         xvar = torch.var(x.reshape(-1, self.block_size), axis=1)

#         self.s = (2 * torch.sqrt(self.zeta * xvar)) / (self.N-1)
#         self.z = - self.s * (self.N/2 - 1) + xmean
#         #self.lk = self.s * self.k_list + self.z
    
#     def fit_uniform_iterative(self, x, verbose=False):

#         self.fit_uniform_normal(x)

#         nblocks = len(x) // self.block_size

#         for i in range(nblocks):

#             xb = x[i*self.block_size:(i+1)*self.block_size]
#             s,z = self.s[i], self.z[i]

#             denum_z = len(xb)
#             sz_prev = torch.zeros(2)
#             for _ in range(2000):

#                 xint = self.quant_uniform(xb, s, z)

#                 sz = torch.tensor([s, z])

#                 num_s = torch.sum((xb - z) * xint)
#                 denum_s = torch.sum(xint**2)
#                 num_z = torch.sum(xb - s * xint)

#                 s = num_s / denum_s
#                 z = num_z / denum_z

#                 self.s[i] = s
#                 self.z[i] = z
                
#                 #self.lk = self.s * self.k_list + self.z

#                 if torch.sum(torch.abs(sz_prev - sz)) / torch.sum(torch.abs(sz)) < 1e-5:
#                     #print('Loss', err)
#                     return
#                 sz_prev = sz
#             print('NOT CONVERGED !!!')
#             print(sz)

#     def fit_nonuniform_quantile(self, x):

#         nblocks = len(x) // self.block_size

#         x_sorted = torch.sort(x.reshape(-1, self.block_size), axis=1)[0]
#         k = torch.arange(0, self.N)
#         ind = ((self.block_size-1) * k.to(torch.float64) / (self.N-1)).to(torch.int)
#         self.tk = x_sorted[:, ind]
#         self.tk[:,0] = torch.nan_to_num(torch.tensor(-float('inf')))
#         self.tk[:,-1] = torch.nan_to_num(torch.tensor(float('inf')))
#         self.lk = torch.zeros(nblocks, self.N-1)
#         for i in k-1:
#             self.lk[:,i] = torch.mean(x_sorted[:,ind[i]:ind[i+1]], axis=1)

#     def fit_nonuniform_analytic(self, x):

#         if not hasattr(self, 'lkopt'):
#             tk = torch.quantile(torch.randn(1000), torch.arange(self.N)/(self.N-1))
#             tk[0] = torch.nan_to_num(torch.tensor(-float('inf')))
#             tk[-1] = torch.nan_to_num(torch.tensor(float('inf')))
#             pdf = torch.distributions.normal.Normal(0., 1.)
#             tk = tk.to(torch.float64)

#             for _ in range(500):

#                 F = self.gauss_cdf(tk, 0., 1.)
#                 P = torch.exp(pdf.log_prob(torch.Tensor(tk)))
                
#                 lk = -(P[1:] - P[:-1]) / (F[1:] - F[:-1])
#                 tk[1:-1] = (lk[1:] + lk[0:-1])/2

#             self.lkopt = lk.to(torch.float32)
#             self.tkopt = tk.to(torch.float32)

#         xmean = torch.mean(x.reshape(-1, self.block_size), axis=1)
#         xstd = torch.std(x.reshape(-1, self.block_size), axis=1)

#         self.lk = (xstd[:,None] * self.lkopt + xmean[:,None])
#         #self.tk = x.mean()


#     def fit_nonuniform_iterative(self, x, verbose=False):

#         self.fit_nonuniform_analytic(x)
#         #self.fit_uniform_minmax(x)

#         if not hasattr(self, 'f'):
#             self.f = torch.ones(x.shape)

#         nblocks = len(x) // self.block_size

#         for i in range(nblocks):

#             xb = x[i*self.block_size:(i+1)*self.block_size]
#             lk = self.lk[i]

#             kmeans = KMeans(n_clusters=self.N-1, init=lk.reshape(-1,1), max_iter=500, tol=1e-5)
#             kmeans.fit(xb.reshape(-1,1), self.f, )
#             self.lk[i] = torch.tensor(kmeans.cluster_centers_[:,0])

#     def fit(self, x, alg):

#         self.block_size=x.numel()

#         if self.qtype=='nonuniform':
            
#             if alg == 'iterative':
#                 self.fit_nonuniform_iterative(x.flatten())
#             elif alg == 'quantile':
#                 self.fit_nonuniform_quantile(x.flatten())
#             elif alg == 'snr':
#                 self.fit_nonuniform_analytic(x.flatten())

#         elif self.qtype=='uniform':
#             if alg == 'iterative':
#                 self.fit_uniform_iterative(x.flatten())
#             elif alg == 'minmax':
#                 self.fit_uniform_minmax(x.flatten())
#             elif alg == 'snr':
#                 self.fit_uniform_normal(x.flatten())

#         elif self.qtype=='float':
#             if alg == 'iterative':
#                 self.fit_float_iterative(x.flatten())
#             elif alg == 'minmax':
#                 self.fit_float_minmax(x.flatten())
#             elif alg == 'snr':
#                 self.fit_float_normal(x.flatten())
#             elif alg == 'cast':
#                 self.fit_float_cast(x.flatten())

#     def quant(self, x, s=1., z=0., lk=None):

#         if self.qtype=='nonuniform':
#             x = self.quant_nonuniform(x, self.lk)
#         elif self.qtype=='uniform':
#             x = self.quant_uniform(x, self.s, self.z)
#         elif self.qtype=='float':
#             x = self.quant_float(x, self.s, self.z)

#         return x
    
#     def fit_and_quant(self, x, alg, s=None, z=None):

#         self.fit(x, alg)
#         if s is None:
#             x = self.quant(x, self.s, self.z)
#         else:
#             x = self.quant(x, s, z)
#         s = self.s
#         z = self.z

#         return x.to(torch.float32), s, z

#     def dequant(self, x, s=1., z=0.,lk=None):

#         if self.qtype=='nonuniform':
#             x = self.dequant_nonuniform(x, lk)

#         elif self.qtype=='uniform':
#             x = self.dequant_uniform(x, s, z)

#         elif self.qtype=='float':
#             x = self.dequant_float(x, s, z)

#         return x


