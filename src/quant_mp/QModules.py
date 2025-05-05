
from typing import Optional, Tuple
import torch
from torch.autograd import Function
import torch.nn as nn
from quant_mp.config import qconfig, rconfig
from quant_mp.quantizer import quantizer, quantizer_base
from torch.nn.functional import linear, conv2d, conv_transpose2d
from math import prod
from quant_mp.lsq import LsqBinaryTernaryExtension

def init_lsq(module):

    if module.rconfig.weight.alg == 'lsq':
        module.weight_clip_val = torch.nn.Parameter(torch.Tensor(1))
        xmax = torch.max(torch.abs(module.weight))
        if module.rconfig.weight.qtype == 'uniform':
            maxq = 2 ** (module.rconfig.weight.qbits - 1) - 1
        elif module.rconfig.weight.qtype == 'float' and module.rconfig.weight.format=='e2m1':
            maxq = 6
        elif module.rconfig.weight.qtype == 'float' and module.rconfig.weight.format=='e3m0':
            maxq = 32
        else:
            raise NotImplementedError(f"Weight config not implemented for LSQ with weight quant {module.rconfig.weight}")

        scale = xmax / maxq
        module.weight_clip_val.data.copy_(scale)

def init_lsq_act(model, input, optimizer):
    
    for name, module in model.named_children():
        if isinstance(module, QLinear):
            if hasattr(module.rconfig, 'activation') and module.rconfig.activation.alg == 'lsq':
                module.activation_clip_val = torch.nn.Parameter(torch.tensor(1., device=input.device))
                xmax = torch.max(torch.abs(input))
                if module.rconfig.weight.qtype == 'uniform':
                    maxq = 2 ** (module.rconfig.weight.qbits - 1) - 1
                elif module.rconfig.weight.qtype == 'float' and module.rconfig.weight.format=='e2m1':
                    maxq = 6
                elif module.rconfig.weight.qtype == 'float' and module.rconfig.weight.format=='e3m0':
                    maxq = 32
                else:
                    raise NotImplementedError(f"Weight config not implemented for LSQ with weight quant {module.rconfig.weight}")

                scale = xmax / maxq
                module.activation_clip_val.data.copy_(scale)
                optimizer.add_param_group({"params": module.activation_clip_val})

def quantizer_tensor(qconfig: qconfig):

    if qconfig.qtype:
        return quantizer(qconfig=qconfig)
    return None

def step_quantizer_delayed(tensor, quantizer: Optional[quantizer_base]):

    if quantizer:

        # Delayed scaling does not work when block-wise
        delayed_scaling = False
        if delayed_scaling:
            params_prev = quantizer.params
            tensor, params, mask = quantizer.fit_and_quant(tensor, params_prev) # Currently fake
            params_new = params_prev if params_prev else params
            #quantizer.s = quantizer.qconfig['beta'] * scale_prev + (1-quantizer.qconfig['beta']) * s if scale_prev else s

        else:
            tensor, params_new, mask = quantizer.fit_and_quant(tensor, None)

        return tensor, params_new[0], mask
    
    return tensor, torch.tensor(1.), torch.ones_like(tensor)
    

class QLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, qweight=None, qact=None, qgrad=None):

        scale_bw = torch.tensor([1., 1.])
        weight, scale_bw[0], wmask = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1], amask = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.))
            scale_bw[0] = 1.

        output = scale_bw[1] * linear(input, weight, None) * scale_bw[0].T
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, wmask, amask)
        ctx.qgrad = qgrad
        ctx.scale_bw = scale_bw

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Currently, grad quant does not support low-precision matmul and block-wise

        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        dtype = grad_output.dtype
        input, weight, bias, wmask, amask = ctx.saved_tensors
        qgrad = ctx.qgrad
        scales = ctx.scale_bw
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        grad_output, scale, _ = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(grad_output, (weight * scales[0]).to(dtype)) * scale * amask

        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.transpose(-1, -2), (input * scales[1]).to(dtype))  * scale * wmask

        return grad_input, grad_weight, grad_bias, None, None, None
    
class QLinear(nn.Module):
    def __init__(self, input_features: int, output_features: int, rconfig: rconfig, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.rconfig = rconfig

        self.qweight = quantizer_tensor(qconfig=self.rconfig.weight)
        self.qact = quantizer_tensor(qconfig=self.rconfig.activation)
        self.qgrad = quantizer_tensor(qconfig=self.rconfig.grad)

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)

        k = torch.sqrt(1. / torch.tensor(input_features))
        nn.init.uniform_(self.weight, -k, k)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)
    
        init_lsq(self)

    def forward(self, input):
        
        if self.rconfig.weight.alg == 'lsq':
            weight = LsqBinaryTernaryExtension.apply(
                self.weight,
                self.weight_clip_val,
                self.qweight,
            )
            if self.rconfig.activation.alg == 'lsq':
                input = LsqBinaryTernaryExtension.apply(
                    input,
                    self.activation_clip_val,
                    self.qact,
                )
            out = nn.functional.linear(input, weight)
            if self.bias is not None:
                out += self.bias.view(1, -1).expand_as(out)
            return out
        
        return QLinearFunction.apply(input, self.weight, self.bias, self.qweight, self.qact, self.qgrad)


class QConv2dFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, qweight=None, qact=None, qgrad=None):

        scale_bw = torch.tensor([1., 1.])
        weight, scale_bw[0], wmask = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1], amask = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.))
            scale_bw[0] = 1.

        if not qweight and qact:
            input = qact.dequant(input, (scale_bw[1], 0.))
            scale_bw[1] = 1.


        output = scale_bw[1] * conv2d(input, weight, None, stride, padding, dilation, groups) * scale_bw[0].T
        if bias is not None:
            output += bias[None,:,None,None]

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
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        grad_output, scale, _ = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)  * scales[0] * scale * amask

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) * scales[1] * scale * wmask

        return grad_input, grad_weight, grad_bias, None, None, None, None,  None,  None,  None

class QConv2d(nn.Module):
    def __init__(self, rconfig: rconfig, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.kernel_size = kernel_size

        self.rconfig = rconfig

        self.qweight = quantizer_tensor(qconfig=rconfig.weight)
        self.qact = quantizer_tensor(qconfig=rconfig.activation)
        self.qgrad = quantizer_tensor(qconfig=rconfig.grad)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)


        k = torch.sqrt(groups / (in_channels * torch.tensor(prod(kernel_size))))
        nn.init.uniform_(self.weight, -k, k)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)
            #nn.init.zeros_(self.bias)


    def forward(self, input):
        return QConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups, self.qweight, self.qact, self.qgrad)
