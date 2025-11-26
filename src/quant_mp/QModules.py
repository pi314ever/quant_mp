"""Quantization-aware building blocks used throughout quant_mp.

This module wires quantization configs and algorithms into drop-in PyTorch
modules. ``QLinear`` and ``QConv2d`` mirror their nn equivalents while
optionally quantizing weights and activations according to ``QuantModuleConfig``.
Helper functions encapsulate quantize/dequantize passes and initialization of
scaling parameters.
"""

from math import prod
from typing import Callable, Optional  # pyright: ignore[reportDeprecated]

import torch
import torch.nn as nn
from loguru import logger
from torch.autograd import Function

from quant_mp.algs.minmax import MinMax
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import DataFormat
from quant_mp.quantizer import dequant, quant


class QuantFunction(Function):
    """Autograd wrapper for quantize/dequantize that keeps gradients flowing."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        quant_config: QuantConfig,
    ):
        output, mask = quant(quant_config.qval_data_format, input, scale, shift)
        output = dequant(output, scale, shift)
        # Avoid saving None in autograd context
        if shift is None:
            ctx.has_shift = False  # type: ignore[attr-defined]
            ctx.save_for_backward(input, scale, mask)
        else:
            ctx.has_shift = True  # type: ignore[attr-defined]
            ctx.save_for_backward(input, scale, shift, mask)
        ctx.quant_config = quant_config
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        quant_config: QuantConfig = ctx.quant_config
        assert quant_config.algorithm is not None
        saved = ctx.saved_tensors
        if getattr(ctx, "has_shift", False):
            input, scale, shift, mask = saved
        else:
            input, scale, mask = saved
            shift = None
        grad_output = grad_outputs[0]
        grad_input, grad_scale, grad_shift = quant_config.algorithm.compute_gradients(
            ctx,
            quant_config.qval_data_format,
            input,
            scale,
            shift,
            mask,
            grad_output,
        )

        return grad_input, grad_scale, grad_shift, None


def quantize_tensor_process(
    full_tensor: torch.Tensor,  # pyright: ignore[reportRedeclaration]
    scale: torch.Tensor,
    shift: torch.Tensor | None,
    in_place_update_fn: Callable[[torch.Tensor, torch.Tensor | None], None],
    quant_config: QuantConfig,
    device: torch.types.Device,
    num_blocks: int,
    block_size: int,
    training: bool,
):
    """
    Quantize ``full_tensor`` using the provided quantization configuration.

    Handles lazy MinMax initialization, optional fit-time parameter updates, and
    dispatches into ``QuantFunction`` for differentiable quantization. Restores
    the original shape and returns a contiguous tensor.

    Args:
        full_tensor: Tensor to quantize; flattened into ``num_blocks`` chunks shaped ``[num_blocks, block_size]``.
        scale: Per-block or tensor-wide scale tensor shaped ``[num_blocks, 1]`` (may start as NaN to trigger init).
        shift: Per-block/tensor shift tensor shaped ``[num_blocks, 1]`` or ``None`` for symmetric quantization.
        in_place_update_fn: Callback used to copy back fitted scale/shift values.
        quant_config: Quantization settings including algorithm and data format.
        device: Target device for quantization operations.
        num_blocks: Number of quantization blocks to reshape into.
        block_size: Number of elements per block after reshaping.
        training: Whether to run algorithm-specific fitting during training.
    """
    assert quant_config.algorithm is not None, (
        "Quantizing tensor must have algorithm specified"
    )
    orig_shape = full_tensor.shape
    full_tensor = full_tensor.reshape(num_blocks, block_size)
    scale = scale.to(device=device, dtype=full_tensor.dtype)
    if shift is not None:
        shift = shift.to(device=device, dtype=full_tensor.dtype)

    # NOTE: Initialization on first run
    if torch.any(torch.isnan(scale)).item():
        init_alg = _resolve_init_algorithm(quant_config)
        assert init_alg.has_fit_params, (
            "Initialization algorithm must support fit_params"
        )
        with torch.no_grad():
            scale, shift = init_alg.fit_params(
                quant_config.qval_data_format, full_tensor, scale, shift
            )
            in_place_update_fn(scale, shift)

    if training and quant_config.algorithm.has_fit_params:
        with torch.no_grad():
            scale, shift = quant_config.algorithm.fit_params(
                quant_config.qval_data_format, full_tensor, scale, shift
            )
            in_place_update_fn(scale, shift)

    full_tensor: torch.Tensor = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
        full_tensor,
        scale,
        shift,
        quant_config,
    )

    return full_tensor.view(orig_shape).contiguous()


def init_qparams_minmax(
    data_format: DataFormat,
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
):
    """
    One-time initialization of quantization parameters using MinMax as proxy.

    Args:
        data_format: Quantized value format (bit width, symmetric/asymmetric, etc.).
        input: Tensor used to derive initial scale/shift statistics; same shape as the block-flattened input to quantization.
        scale: Scale tensor shaped ``[num_blocks, 1]`` to be updated in-place.
        shift: Shift tensor shaped ``[num_blocks, 1]`` to be updated or ``None`` when symmetric.
    """
    return MinMax().fit_params(data_format, input, scale, shift)


def _resolve_init_algorithm(quant_config: QuantConfig):
    """
    Select the initialization algorithm for quantization parameters.

    Prefers an explicitly provided ``init_algorithm``. If absent, uses the
    primary algorithm when it supports ``fit_params``. Falls back to MinMax for
    algorithms without parameter fitting (e.g., LSQ).
    """
    if quant_config.init_algorithm is not None:
        if not quant_config.init_algorithm.has_fit_params:
            raise ValueError(
                "Invalid init algorithm: must implement fit_params for initialization"
            )
        return quant_config.init_algorithm

    assert quant_config.algorithm is not None
    if quant_config.algorithm.has_fit_params:
        return quant_config.algorithm

    return MinMax()


class QModuleMixin(object):
    """Quantization Module mixin for quantization functionality on weights and inputs."""

    def __init__(
        self,
        block_size: Optional[int] = None,
        num_blocks: Optional[int] = None,
        qmodule_config: Optional[QuantModuleConfig] = None,
        device=None,
        dtype=None,
    ):
        self.device = device
        self.dtype = dtype
        self.config = qmodule_config
        if qmodule_config is None:
            return
        assert block_size is not None and num_blocks is not None

        self.block_size = block_size
        self.num_blocks = num_blocks
        self._maybe_init_weight_qconfig()
        self._maybe_init_activation_qconfig()

    def _maybe_init_weight_qconfig(self):
        if self.config is None:
            return
        qconfig = self.config.weight
        if qconfig is None:
            logger.trace("Skipping empty weight qconfig")
            return

        logger.trace(f"Configuring weight quantizer {qconfig=}")
        if qconfig.algorithm is None:
            msg = "Invalid qmodule config: Must have weight quant algorithm set."
            logger.error(msg)
            raise ValueError(msg)

        self.weight_qconfig = qconfig
        self.weight_alg = qconfig.algorithm

        weight_scale = (
            torch.ones(self.num_blocks)
            .reshape(self.num_blocks, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        # NOTE: Nan used to trigger init
        weight_scale[0] = float("nan")
        if self.weight_qconfig.symmetric:
            weight_shift = None
        else:
            weight_shift = (
                torch.zeros(self.num_blocks)
                .reshape(self.num_blocks, 1)
                .to(device=self.device, dtype=self.dtype)
            )

        requires_grad = not self.weight_alg.has_fit_params
        self.weight_scale = torch.nn.Parameter(
            weight_scale, requires_grad=requires_grad
        )
        if weight_shift is not None:
            self.weight_shift = torch.nn.Parameter(
                weight_shift, requires_grad=requires_grad
            )

    def _maybe_init_activation_qconfig(self):
        if self.config is None:
            return
        qconfig = self.config.activation
        if qconfig is None:
            logger.trace("Skipping empty activation qconfig")
            return
        logger.trace(f"Configuring activation quantizer {qconfig=}")
        if qconfig.algorithm is None:
            msg = "Invalid qmodule config: Must have activation quant algorithm set if activation quantconfig exists."
            logger.error(msg)
            raise ValueError(msg)
        if qconfig.qblock_size is not None:
            msg = "Invalid qmodule config: Activation qconfig must be tensor-wise."
            logger.error(msg)
            raise ValueError(msg)
        self.activation_qconfig = qconfig
        self.activation_alg = qconfig.algorithm
        activation_scale = torch.tensor([float("nan")]).to(
            device=self.device, dtype=self.dtype
        )
        # NOTE: Activation shift values are zeroed for forced symmetric quant
        requires_grad = not self.activation_alg.has_fit_params
        self.activation_scale = torch.nn.Parameter(
            activation_scale, requires_grad=requires_grad
        )
        if not self.activation_qconfig.symmetric:
            self.activation_shift = torch.nn.Parameter(
                torch.zeros_like(activation_scale).to(
                    device=self.device, dtype=self.dtype
                ),
                requires_grad=requires_grad,
            )

    def _maybe_quantize_weight(self, weight, device, training):
        weight = weight.to(device)
        if self.config is not None and self.config.weight is not None:
            weight = quantize_tensor_process(
                weight,
                self.weight_scale,
                self.weight_shift if not self.weight_qconfig.symmetric else None,
                self._update_weight_qparams,
                self.weight_qconfig,
                device,
                self.num_blocks,
                self.block_size,
                training,
            )
        return weight

    def _maybe_quantize_input(self, input, device, training):
        if self.config is not None and self.config.activation is not None:
            input = quantize_tensor_process(
                input,
                self.activation_scale,
                self.activation_shift
                if not self.activation_qconfig.symmetric
                else None,
                self._update_activation_qparams,
                self.activation_qconfig,
                device,
                1,
                input.numel(),
                training,
            )
        return input

    def _update_weight_qparams(self, scale: torch.Tensor, shift: torch.Tensor | None):
        with torch.no_grad():
            _ = self.weight_scale.copy_(scale)
            if not self.weight_qconfig.symmetric:
                assert shift is not None
                _ = self.weight_shift.copy_(shift)

    def _update_activation_qparams(
        self, scale: torch.Tensor, shift: torch.Tensor | None
    ):
        with torch.no_grad():
            # Manually update scale and shift
            _ = self.activation_scale.copy_(scale.squeeze())
            if not self.activation_qconfig.symmetric:
                assert shift is not None
                _ = self.activation_shift.copy_(shift.squeeze())


class QLinear(QModuleMixin, nn.Linear):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Linear module that applies block- or tensor-wise quantization when configured."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=True,
        device=None,
        dtype=None,
        qlinear_config: Optional[QuantModuleConfig] = None,
    ):
        """
        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            bias: Whether to include a bias term.
            device: Torch device for parameter initialization.
            dtype: Torch dtype for parameter initialization.
            qlinear_config: Optional quantization configuration for weights/activations.
        """
        nn.Linear.__init__(self, in_features, out_features, bias, device, dtype)
        block_size, num_blocks = self._resolve_block_size(
            qlinear_config, in_features, out_features
        )
        logger.trace(f"Initializing QLinear with quant config: {qlinear_config}")
        self.config = qlinear_config
        QModuleMixin.__init__(
            self, block_size, num_blocks, qlinear_config, device, dtype
        )

    def _resolve_block_size(
        self, qconfig: QuantModuleConfig | None, in_features: int, out_features: int
    ) -> tuple[int, int] | tuple[None, None]:
        if qconfig is None or qconfig.weight is None:
            return None, None

        if qconfig.weight.qblock_size is None:
            block_size = out_features * in_features
        elif isinstance(qconfig.weight.qblock_size, int):
            block_size = qconfig.weight.qblock_size
            assert out_features * in_features % block_size == 0, (
                f"Linear dimensions ({out_features} x {in_features}) is not divisible by block size of {block_size}"
            )
        elif qconfig.weight.qblock_size == "channel":
            block_size = out_features
        else:
            raise ValueError(f"Unsupported block size {qconfig.weight.qblock_size}.")
        num_blocks = out_features * in_features // block_size
        logger.trace(
            f"Config weight block size: {qconfig.weight.qblock_size} => {in_features=} {out_features=} {num_blocks=} {block_size=}"
        )
        return block_size, num_blocks

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply optional activation and weight quantization before matmul."""
        device = input.device
        weight = self._maybe_quantize_weight(self.weight, device, self.training)
        input = self._maybe_quantize_input(input, device, self.training)

        out = nn.functional.linear(
            input, weight, None if self.bias is None else self.bias.to(input.device)
        )
        return out


class QConv2d(QModuleMixin, nn.Conv2d):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """2D convolution that mirrors ``nn.Conv2d`` with optional quantized weights/activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        qconv_config: Optional[QuantModuleConfig] = None,
    ):
        """
        Args:
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            stride: Stride of the convolution.
            padding: Zero-padding added to both sides of the input.
            dilation: Spacing between kernel elements.
            groups: Number of blocked connections from input to output channels.
            bias: Whether to include a bias term.
            padding_mode: Padding mode (e.g., ``"zeros"``, ``"reflect"``).
            qconv_config: Optional quantization configuration for weights/activations.
        """
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        logger.trace(f"Initializing QConv2d with quant config: {qconv_config}")
        self.config = qconv_config
        block_size, num_blocks = self._resolve_block_size()
        QModuleMixin.__init__(self, block_size, num_blocks, qconv_config, device, dtype)

    def _resolve_block_size(self) -> tuple[int, int] | tuple[None, None]:
        qconfig = self.config
        in_channels = self.in_channels
        out_channels = self.out_channels

        if qconfig is None or qconfig.weight is None:
            return None, None
        if qconfig.weight.qblock_size is None:
            block_size = in_channels * out_channels * prod(self.kernel_size)
        elif isinstance(qconfig.weight.qblock_size, int):
            block_size = qconfig.weight.qblock_size
            assert prod(self.weight.shape) % block_size == 0, (
                f"Conv dimensions ({self.weight.shape}) is not divisible by block size of {block_size}"
            )
        elif qconfig.weight.qblock_size == "channel":
            block_size = out_channels
        else:
            raise ValueError(f"Unsupported block size {qconfig.weight.qblock_size}.")
        num_blocks = prod(self.weight.shape) // block_size
        return block_size, num_blocks

    def forward(self, input: torch.Tensor):
        """Apply optional activation and weight quantization before convolution."""
        device = input.device
        weight = self._maybe_quantize_weight(self.weight, device, self.training)
        input = self._maybe_quantize_input(input, device, self.training)

        return self._conv_forward(input, weight, self.bias)
