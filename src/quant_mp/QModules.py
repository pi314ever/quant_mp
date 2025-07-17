from math import prod
from typing import Optional, Tuple

from loguru import logger
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.functional import conv2d

from quant_mp.algs.minmax import MinMax
from quant_mp.config import QuantConfig, QuantLinearConfig
from quant_mp.datatypes.template import DataFormat
from quant_mp.quantizer import quant, dequant


class QuantFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        quant_config: QuantConfig,
    ):
        output, mask = quant(quant_config.qval_data_format, input, scale, shift)
        output = dequant(output, scale, shift)
        ctx.save_for_backward(input, scale, shift, mask)
        ctx.quant_config = quant_config
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        quant_config: QuantConfig = ctx.quant_config
        assert quant_config.algorithm is not None
        input, scale, shift, mask = ctx.saved_tensors
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

        return grad_input, grad_scale, grad_shift, None, None


def init_activation_minmax(
    data_format: DataFormat,
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
):
    """One-time activation of activation quantization using minmax as proxy"""
    return MinMax().fit_params(data_format, input, scale, shift)


class QLinear(nn.Linear):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        qlinear_config: Optional[QuantLinearConfig] = None,
        bias=True,
    ):
        super().__init__(input_features, output_features, bias=bias)
        self.config = qlinear_config

        if qlinear_config is not None and qlinear_config.weight is not None:
            if qlinear_config.weight.algorithm is None:
                msg = "Invalid qlinear config: Must have weight quant algorithm set."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            self.weight_qconfig = qlinear_config.weight
            self.weight_alg = qlinear_config.weight.algorithm

            # Initialize params
            if qlinear_config.weight.qblock_size is None:
                block_size = output_features * input_features
            elif isinstance(qlinear_config.weight.qblock_size, int):
                block_size = qlinear_config.weight.qblock_size
                assert output_features * input_features % block_size == 0, (
                    f"Linear dimensions ({output_features} x {input_features}) are not divisible by block size of {block_size}"
                )
            elif qlinear_config.weight.qblock_size == "channel":
                block_size = output_features
            else:
                raise ValueError(
                    f"Unsupported block size {qlinear_config.weight.qblock_size}."
                )
            num_blocks = output_features * input_features // block_size
            self.block_size = block_size
            self.num_blocks = num_blocks

            # NOTE: Minmax usage here may need to change.
            weight_scale = torch.ones(num_blocks).reshape(num_blocks, 1)
            if self.weight_qconfig.symmetric:
                weight_shift = None
            else:
                weight_shift = torch.zeros(num_blocks).reshape(num_blocks, 1)

            with torch.no_grad():
                weight_scale, weight_shift = MinMax().fit_params(
                    self.weight_qconfig.qval_data_format,
                    self.weight.view(num_blocks, block_size),
                    weight_scale,
                    weight_shift,
                )
            self.weight_scale = torch.nn.Parameter(weight_scale)
            if weight_shift is not None:
                self.weight_shift = torch.nn.Parameter(weight_shift)

        if qlinear_config is not None and qlinear_config.activation is not None:
            if qlinear_config.activation.algorithm is None:
                msg = "Invalid qlinear config: Must have activation quant algorithm set if activation quantconfig exists."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            if qlinear_config.activation.qblock_size is not None:
                msg = "Invalid qlinear config: Activation qconfig must be tensor-wise."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            self.activation_qconfig = qlinear_config.activation
            self.activation_alg = qlinear_config.activation.algorithm
            activation_scale = torch.tensor([float("nan")])
            # NOTE: Activation shift values are zeroed for forced symmetric quant
            self.activation_scale = torch.nn.Parameter(activation_scale)
            if not self.activation_qconfig.symmetric:
                self.activation_shift = torch.nn.Parameter(
                    torch.zeros_like(activation_scale)
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        weight = self.weight.to(device)
        if self.config is not None and self.config.weight is not None:
            orig_shape = self.weight.shape
            weight = weight.view(self.num_blocks, self.block_size)
            scale = self.weight_scale.to(device)
            shift = (
                None if self.weight_qconfig.symmetric else self.weight_shift.to(device)
            )
            if self.weight_alg.has_fit_params:
                scale, shift = self.weight_alg.fit_params(
                    self.weight_qconfig.qval_data_format, weight, scale, shift
                )
                # Manually update scale and shift
                _ = self.weight_scale.data.copy_(scale)
                if not self.weight_qconfig.symmetric:
                    assert shift is not None
                    _ = self.weight_shift.data.copy_(shift)

            weight: torch.Tensor = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                weight,
                scale,
                shift,
                self.weight_qconfig,
            )

            weight = weight.view(orig_shape)

        if self.config is not None and self.config.activation is not None:
            input_orig_shape = input.shape
            input = input.view(1, -1)
            scale = self.activation_scale.to(device)
            shift = (
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device)
            )

            # NOTE:MinMax activation on first run
            if torch.any(torch.isnan(self.activation_scale)).item():
                with torch.no_grad():
                    scale, shift = init_activation_minmax(
                        self.activation_qconfig.qval_data_format, input, scale, shift
                    )
                # Manually update scale and shift
                _ = self.activation_scale.data.copy_(scale)
                if not self.weight_qconfig.symmetric:
                    assert shift is not None
                    _ = self.activation_shift.data.copy_(shift)

            if self.activation_alg.has_fit_params:
                scale, shift = self.activation_alg.fit_params(
                    self.activation_qconfig.qval_data_format, input, scale, shift
                )
                # Manually update scale and shift
                _ = self.activation_scale.data.copy_(scale)
                if not self.weight_qconfig.symmetric:
                    assert shift is not None
                    _ = self.activation_shift.data.copy_(shift)

            input = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                input,
                self.activation_scale.to(device),
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device),
                self.activation_qconfig,
            )

            input = input.view(input_orig_shape)

        out = nn.functional.linear(input, weight)
        if self.bias is not None:
            out += self.bias.unsqueeze(0).expand_as(out).to(input.device)
        return out


# FIXME: Conv2D function not converted to new API
class QConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        qweight=None,
        qact=None,
        qgrad=None,
    ):
        scale_bw = torch.tensor([1.0, 1.0])
        weight, scale_bw[0], wmask = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1], amask = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.0))
            scale_bw[0] = 1.0

        if not qweight and qact:
            input = qact.dequant(input, (scale_bw[1], 0.0))
            scale_bw[1] = 1.0

        output = (
            scale_bw[1]
            * conv2d(input, weight, None, stride, padding, dilation, groups)
            * scale_bw[0].T
        )
        if bias is not None:
            output += bias[None, :, None, None]

        ctx.save_for_backward(input, weight, bias, wmask, amask)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        ctx.qgrad = qgrad
        ctx.scale_bw = scale_bw

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        input, weight, bias, wmask, amask = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        qgrad = ctx.qgrad
        scales = ctx.scale_bw
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        grad_output, scale, _ = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = (
                torch.nn.grad.conv2d_input(
                    input.shape, weight, grad_output, stride, padding, dilation, groups
                )
                * scales[0]
                * scale
                * amask
            )

        if ctx.needs_input_grad[1]:
            grad_weight = (
                torch.nn.grad.conv2d_weight(
                    input, weight.shape, grad_output, stride, padding, dilation, groups
                )
                * scales[1]
                * scale
                * wmask
            )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# TODO: Update to new architecture
class QConv2d(nn.Module):
    def __init__(
        self,
        rconfig: QuantLinearConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        self.rconfig = rconfig

        self.quantizer_weight = None
        if rconfig.weight is not None:
            if rconfig.weight.qblock_size is None:
                block_size = (
                    in_channels * out_channels * kernel_size[0] * kernel_size[1]
                )
            elif isinstance(rconfig.weight.qblock_size, int):
                block_size = rconfig.weight.qblock_size
            elif rconfig.weight.qblock_size == "channel":
                # FIXME: Figure out what this needs to be
                block_size = ...  # output_features
            else:
                raise ValueError(
                    f"Unsupported block size {rconfig.weight.qblock_size}."
                )
            # FIXME: Same here
            num_blocks = ...  # output_features * input_features // block_size
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.quantizer_weight = get_quantizer(
                qconfig=self.rconfig.weight, device=device
            )
            self.quantize_weight_cls = get_quantize_function_cls(self.rconfig.weight)
            # Initialize weight and shift val using minmax
            assert isinstance(self.quantizer_weight, (FloatQuantizer, UniformQuantizer))
            with torch.no_grad():
                # FIXME: Fix this usage
                weight_clip_val, weight_shift_val = self.quantizer_weight.fit_minmax(
                    self.weight.view(num_blocks, block_size),
                    torch.ones(num_blocks),
                    torch.zeros(num_blocks),
                )
            if rconfig.weight.alg_requires_grad_params:
                self.weight_clip_val = torch.nn.Parameter(weight_clip_val)
                self.weight_shift_val = torch.nn.Parameter(weight_shift_val)
            else:
                self.register_buffer("weight_clip_val", weight_clip_val)
                self.register_buffer("weight_shift_val", weight_shift_val)

        if rconfig.activation is not None:
            self.quantizer_act = get_quantizer(
                qconfig=self.rconfig.activation, device=device
            )
            if rconfig.activation.qblock_size is not None:
                print(
                    f"Block size ({self.rconfig.activation.qblock_size}) is not supported for activations. Tensor-wise quantization will be applied"
                )
            self.quantize_act_cls = get_quantize_function_cls(self.rconfig.activation)
            activation_clip_val = torch.tensor([float("nan")])
            # NOTE: Activation shift values are zeroed for forced symmetric quant
            activation_shift_val = torch.zeros_like(activation_clip_val, device=device)
            if rconfig.activation.alg_requires_grad_params:
                self.activation_clip_val = torch.nn.Parameter(activation_clip_val)
                self.activation_shift_val = torch.nn.Parameter(activation_shift_val)
            else:
                self.register_buffer("activation_clip_val", activation_clip_val)
                self.register_buffer("activation_shift_val", activation_shift_val)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        k = torch.sqrt(groups / (in_channels * torch.tensor(prod(kernel_size))))
        nn.init.uniform_(self.weight, -k, k)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)
            # nn.init.zeros_(self.bias)

    def forward(self, input):
        return QConv2dFunction.apply(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.quantizer_weight,
            self.quantizer_act,
        )
