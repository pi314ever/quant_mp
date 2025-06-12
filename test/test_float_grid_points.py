import pytest
import torch

from quant_mp.config import QuantConfig
from quant_mp.quantizer import FloatQuantizer


@pytest.mark.parametrize("format_fp4", ["e2m1", "e3m0"])
def test_fp4_grid(format_fp4):
    qconfig_ = QuantConfig(qtype="float", qbits=4, alg="cast", format=format_fp4)

    module = FloatQuantizer(qconfig=qconfig_)

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
    qconfig_ = QuantConfig(qtype="float", qbits=8, alg="cast", format=format_fp8)

    module = FloatQuantizer(qconfig=qconfig_)

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
    qconfig_ = QuantConfig(qtype="float", qbits=16, alg="cast", format=format_fp16)

    module = FloatQuantizer(qconfig=qconfig_)

    Gp = torch.hstack(
        [
            torch.arange(module.xr[r], module.xr[r + 1], module.vr[r])
            for r in range(len(module.vr))
        ]
    )
    Gp = torch.concat((Gp, torch.tensor([Gp[-1] + module.vr[-1]])))

    assert torch.equal(Gp, module.G)
