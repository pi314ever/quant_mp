from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from quant_mp.algs.minmax import MinMax
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import get_data_format
from quant_mp.QModules import QLinear


class TestQLinear:
    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def qlinear_config(self):
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
    def qlinear_config_with_activation(self):
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

    def test_qlinear_init_no_config(self):
        # Test initialization without quantization config
        layer = QLinear(10, 20)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 10
        assert layer.out_features == 20
        assert layer.config is None

    def test_qlinear_init_with_weight_config(self, qlinear_config):
        # Test initialization with weight quantization config
        layer = QLinear(10, 20, qlinear_config=qlinear_config)
        assert layer.config is not None
        assert layer.weight_qconfig is not None
        assert layer.weight_alg is not None
        assert hasattr(layer, "weight_scale")
        assert layer.block_size == 20  # Since qblock_size is "channel"
        assert layer.num_blocks == 10  # input_features

    def test_qlinear_init_with_activation_config(self, qlinear_config_with_activation):
        # Test initialization with both weight and activation config
        layer = QLinear(10, 20, qlinear_config=qlinear_config_with_activation)
        assert layer.config is not None
        assert layer.weight_qconfig is not None
        assert layer.activation_qconfig is not None
        assert hasattr(layer, "weight_scale")
        assert hasattr(layer, "activation_scale")
        assert torch.isnan(layer.activation_scale).all()  # Should start as NaN

    def test_qlinear_forward_no_quant(self):
        # Test forward pass without quantization
        layer = QLinear(10, 5)
        x = torch.randn(2, 10)
        out = layer(x)
        assert out.shape == (2, 5)

    def test_qlinear_forward_weight_quant(self, qlinear_config, device):
        # Test forward pass with weight quantization
        layer = QLinear(10, 5, qlinear_config).to(device)
        x = torch.randn(2, 10, device=device)
        out = layer(x)
        assert out.shape == (2, 5)

    def test_qlinear_forward_full_quant(self, qlinear_config_with_activation, device):
        # Test forward pass with both weight and activation quantization
        layer = QLinear(10, 5, qlinear_config=qlinear_config_with_activation).to(device)
        x = torch.randn(2, 10, device=device)

        # First forward pass should initialize activation quantization
        out = layer(x)
        assert out.shape == (2, 5)
        assert not torch.isnan(layer.activation_scale).any()

        # Second forward pass should use the initialized values
        out2 = layer(x)
        assert out2.shape == (2, 5)

    @pytest.mark.parametrize(
        "qblock_size,expected_block_size", [(None, 400), ("channel", 40), (20, 20)]
    )
    def test_qlinear_block_sizes(
        self, qlinear_config, qblock_size, expected_block_size
    ):
        # Test block size handling
        qlinear_config = replace(
            qlinear_config,
            weight=replace(qlinear_config.weight, qblock_size=qblock_size),
        )
        layer = QLinear(10, 40, qlinear_config=qlinear_config)
        assert layer.block_size == expected_block_size
