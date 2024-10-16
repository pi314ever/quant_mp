
import torch
from torch.autograd import Function
import torch.nn as nn
from quant_mp.quantizer import quantizer
from torch.nn.functional import linear, conv2d, conv_transpose2d
from math import prod


def quantizer_tensor(qconfig):

    if qconfig['qtype']:
        return quantizer(qconfig=qconfig)
    return None

def step_quantizer_delayed(tensor, quantizer):

    if quantizer:

        # Delayed scaling does not work when block-wise
        delayed_scaling = False
        if delayed_scaling:
            params_prev = quantizer.params
            tensor, params = quantizer.fit_and_quant(tensor, params_prev) # Currently fake
            params_new = params_prev if params_prev else params
            #quantizer.s = quantizer.qconfig['beta'] * scale_prev + (1-quantizer.qconfig['beta']) * s if scale_prev else s

        else:
            tensor, params_new = quantizer.fit_and_quant(tensor, None)

        return tensor, params_new[0]
    
    return tensor, torch.tensor(1.)

class QLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, qweight=None, qact=None, qgrad=None):

        scale_bw = torch.tensor([1., 1.])
        weight, scale_bw[0] = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1] = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.))
            scale_bw[0] = 1.

        output = scale_bw[1] * linear(input, weight, None) * scale_bw[0].T
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias)
        ctx.qgrad = qgrad
        ctx.scale_bw = scale_bw

        return output

    @staticmethod
    def backward(ctx, grad_output):

        # Currently, grad quant does not support low-precision matmul and block-wise

        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        input, weight, bias = ctx.saved_tensors
        qgrad = ctx.qgrad
        scales = ctx.scale_bw
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        grad_output, scale = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight * scales[0]) * scale

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input * scales[1])  * scale

        return grad_input, grad_weight, grad_bias, None, None, None
    
class QLinear(nn.Module):
    def __init__(self, input_features, output_features, qconfig, bias=True):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.qconfig = qconfig
        self.qconfig['input_features'] = input_features
        self.qconfig['output_features'] = output_features


        self.qweight = quantizer_tensor(qconfig=self.qconfig['weight'])
        self.qact = quantizer_tensor(qconfig=self.qconfig['activation'])
        self.qgrad = quantizer_tensor(qconfig=self.qconfig['grad'])

        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            self.register_parameter('bias', None)

        k = torch.sqrt(1. / torch.tensor(input_features))
        nn.init.uniform_(self.weight, -k, k)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)

    def forward(self, input):
        return QLinearFunction.apply(input, self.weight, self.bias, self.qweight, self.qact, self.qgrad)

class QConv2dFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups, qweight=None, qact=None, qgrad=None):

        scale_bw = torch.tensor([1., 1.])
        weight, scale_bw[0] = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1] = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.))
            scale_bw[0] = 1.

        if not qweight and qact:
            input = qact.dequant(input, (scale_bw[1], 0.))
            scale_bw[1] = 1.


        output = scale_bw[1] * conv2d(input, weight, None, stride, padding, dilation, groups) * scale_bw[0].T
        if bias is not None:
            output += bias[None,:,None,None]

        ctx.save_for_backward(input, weight, bias)
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

        input, weight, bias = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        qgrad = ctx.qgrad
        scales = ctx.scale_bw
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)

        grad_output, scale = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)  * scales[0] * scale

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) * scales[1] * scale

        return grad_input, grad_weight, grad_bias, None, None, None, None,  None,  None,  None

class QConv2d(nn.Module):
    def __init__(self, qconfig, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.groups=groups
        self.kernel_size = kernel_size

        self.qconfig = qconfig

        self.qweight = quantizer_tensor(qconfig=qconfig['weight'])
        self.qact = quantizer_tensor(qconfig=qconfig['activation'])
        self.qgrad = quantizer_tensor(qconfig=qconfig['grad'])

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

    

 
    


