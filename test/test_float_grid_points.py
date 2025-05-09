import torch
import pytest

from quant_mp.quantizer import quantizer_float
from quant_mp.config import qconfig


@pytest.mark.parametrize("format_fp4", ["e2m1", "e3m0"])
def test_fp4_grid(format_fp4):
    m = 2048

    qconfig_ = qconfig(qtype="float", qbits=4, alg="cast", format=format_fp4)

    module = quantizer_float(qconfig=qconfig_)

    X = torch.max(module.G) * torch.randn([m, m])
    x = X.flatten()

    module.fit_and_quant(x)

    Gp = torch.hstack(
        [
            torch.arange(module.xr[r], module.xr[r + 1], module.vr[r])
            for r in range(len(module.vr))
        ]
    )
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)


@pytest.mark.parametrize("format_fp8", ["e4m3", "e5m2"])
def test_fp8_grid(format_fp8):
    m = 2048

    qconfig_ = qconfig(qtype="float", qbits=8, alg="cast", format=format_fp8)

    module = quantizer_float(qconfig=qconfig_)

    X = torch.max(module.G) * torch.randn([m, m])
    x = X.flatten()

    xdeq = module.fit_and_quant(x)

    Gp = torch.hstack(
        [
            torch.arange(module.xr[r], module.xr[r + 1], module.vr[r])
            for r in range(len(module.vr))
        ]
    )
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)


@pytest.mark.parametrize("format_fp16", ["fp"])
def test_fp16_grid(format_fp16):
    m = 2048

    qconfig_ = qconfig(qtype="float", qbits=16, alg="cast", format=format_fp16)

    module = quantizer_float(qconfig=qconfig_)

    X = torch.max(module.G) * torch.randn([m, m])
    x = X.flatten()

    xdeq = module.fit_and_quant(x)

    Gp = torch.hstack(
        [
            torch.arange(module.xr[r], module.xr[r + 1], module.vr[r])
            for r in range(len(module.vr))
        ]
    )
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)
