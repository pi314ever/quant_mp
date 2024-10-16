
import torch
import pytest
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.quantizer_weight import quantizer_weight


@pytest.mark.parametrize("format_fp4", ['e2m1', 'e3m0'])
def test_fp4_grid(format_fp4):
    m = 2048

    module = quantizer_weight(b=4, qtype='float', format_fp4=format_fp4)

    X = torch.max(module.G) * torch.randn([m,m])
    x = X.flatten()

    module.fit_and_quant(x, alg='snr')

    Gp = torch.hstack([torch.arange(module.xr[r], module.xr[r+1], module.vr[r]) for r in range(len(module.vr))])
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)



@pytest.mark.parametrize("format_fp8", ['e4m3', 'e5m2'])
def test_fp8_grid(format_fp8):
    m = 2048

    module = quantizer_weight(b=8, qtype='float', format_fp8=format_fp8)

    X = torch.max(module.G) * torch.randn([m,m])
    x = X.flatten()

    xdeq = module.fit_and_quant(x, alg='snr')

    Gp = torch.hstack([torch.arange(module.xr[r], module.xr[r+1], module.vr[r]) for r in range(len(module.vr))])
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)


@pytest.mark.parametrize("format_fp16", ['fp'])
def test_fp16_grid(format_fp16):
    m = 2048

    module = quantizer_weight(b=16, qtype='float', format_fp16=format_fp16)

    X = torch.max(module.G) * torch.randn([m,m])
    x = X.flatten()

    xdeq = module.fit_and_quant(x, alg='snr')

    Gp = torch.hstack([torch.arange(module.xr[r], module.xr[r+1], module.vr[r]) for r in range(len(module.vr))])
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)