from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from quant_mp.algs.minmax import MinMax
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import get_data_format
from quant_mp.QModules import QConv2d


class TestQConv2d:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def qconv_config(self):
        # Create a simple test configuration with weight quantization only
        weight_config = QuantConfig(
            qval_data_format=get_data_format("int8"),
            qparam_data_format=get_data_format("fp32"),
            symmetric=True,
            qblock_size="channel",
            algorithm=MinMax(),
        )
        return QuantModuleConfig(weight=weight_config, activation=None)

    @pytest.fixture
    def qconv_config_with_activation(self):
        # Configuration with both weight and activation quantization
        weight_config = QuantConfig(
            qval_data_format=get_data_format("int8"),
            qparam_data_format=get_data_format("fp32"),
            symmetric=True,
            qblock_size="channel",
            algorithm=MinMax(),
        )
        activation_config = QuantConfig(
            qval_data_format=get_data_format("int8"),
            qparam_data_format=get_data_format("fp32"),
            symmetric=True,
            qblock_size=None,
            algorithm=MinMax(),
        )
        return QuantModuleConfig(weight=weight_config, activation=activation_config)

    def test_qconv2d_init_no_config(self):
        # Test initialization without quantization config
        layer = QConv2d(3, 16, kernel_size=3)
        assert isinstance(layer, nn.Conv2d)
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == (3, 3)
        assert layer.config is None

    def test_qconv2d_init_with_weight_config(self, qconv_config):
        # Test initialization with weight quantization config
        layer = QConv2d(3, 16, kernel_size=3, qconv_config=qconv_config)
        assert layer.config is not None
        assert layer.weight_qconfig is not None
        assert layer.weight_alg is not None
        assert hasattr(layer, "weight_scale")
        assert layer.block_size == 16  # Since qblock_size is "channel"
        assert layer.num_blocks == 3 * 3 * 3  # in_channels * kernel_size^2

    def test_qconv2d_init_with_activation_config(self, qconv_config_with_activation):
        # Test initialization with both weight and activation config
        layer = QConv2d(3, 16, kernel_size=3, qconv_config=qconv_config_with_activation)
        assert layer.config is not None
        assert layer.weight_qconfig is not None
        assert layer.activation_qconfig is not None
        assert hasattr(layer, "weight_scale")
        assert hasattr(layer, "activation_scale")
        assert torch.isnan(layer.activation_scale).all()  # Should start as NaN

    def test_qconv2d_forward_no_quant(self):
        # Test forward pass without quantization
        layer = QConv2d(3, 16, kernel_size=3)
        x = torch.randn(2, 3, 32, 32)
        out = layer(x)
        assert out.shape == (2, 16, 30, 30)  # Output size with kernel=3, padding=0

    def test_qconv2d_forward_weight_quant(self, qconv_config, device):
        # Test forward pass with weight quantization
        layer = QConv2d(3, 16, kernel_size=3, qconv_config=qconv_config).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)
        out = layer(x)
        assert out.shape == (2, 16, 30, 30)

    def test_qconv2d_forward_full_quant(self, qconv_config_with_activation, device):
        # Test forward pass with both weight and activation quantization
        layer = QConv2d(
            3, 16, kernel_size=3, qconv_config=qconv_config_with_activation
        ).to(device)
        x = torch.randn(2, 3, 32, 32, device=device)

        # First forward pass should initialize activation quantization
        out = layer(x)
        assert out.shape == (2, 16, 30, 30)
        assert not torch.isnan(layer.activation_scale).any()

        # Second forward pass should use the initialized values
        out2 = layer(x)
        assert out2.shape == (2, 16, 30, 30)

    @pytest.mark.parametrize(
        "qblock_size,expected_block_size", [(None, 432), ("channel", 16), (8, 8)]
    )
    def test_qconv2d_block_sizes(self, qconv_config, qblock_size, expected_block_size):
        # Test block size handling
        qconv_config = replace(
            qconv_config,
            weight=replace(qconv_config.weight, qblock_size=qblock_size),
        )
        layer = QConv2d(3, 16, kernel_size=3, qconv_config=qconv_config)
        assert layer.block_size == expected_block_size
