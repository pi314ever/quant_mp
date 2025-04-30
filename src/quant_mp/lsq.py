
import torch
import math


def init_lsq(module):
    module.weight_clip_val = torch.nn.Parameter(torch.Tensor(module.weight.shape[0], 1))
    xmax, _ = torch.max(torch.abs(module.weight), dim=-1, keepdim=True)
    if module.qconfig.weight.qtype == 'uniform':
        maxq = 2 ** (module.qconfig.weight.qbits - 1) - 1
    elif module.qconfig.weight.qtype == 'float' and module.qconfig.weight.format=='e2m1':
        maxq = 6
    elif module.qconfig.weight.qtype == 'float' and module.qconfig.weight.format=='e3m0':
        maxq = 32

    scale = xmax / maxq
    module.weight_clip_val.data.copy_(scale)

class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, qconfig):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :return: quantized output
        """
        ctx.num_bits = qconfig.qbits

        if qconfig.qtype == 'uniform':
            Qn = -(2 ** (qconfig.qbits - 1))
            Qp = 2 ** (qconfig.qbits - 1) - 1
        elif qconfig.qtype == 'float' and qconfig.format=='e2m1':
            Qn = -6
            Qp = 6
        elif qconfig.qtype == 'float' and qconfig.format=='e3m0':
            Qn = -32
            Qp = 32

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp

        if qconfig.qtype == 'uniform':
            q_w = (input / alpha).round().clamp(Qn, Qp)
        elif qconfig.qtype == 'float':
            if qconfig.format=='e2m1':
                M = 1
                bias = 1
            elif qconfig.format=='e3m0':
                M = 0
                bias = 2

            x = torch.clamp(input / alpha, Qn, Qp)
            v = 2**(torch.floor(torch.log2(torch.abs(x))) - M)
            v[torch.floor(torch.log2(torch.abs(x)) + bias) < 1] = 2**(1-M-bias)
            q_w = v * torch.round(x / v)


        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = (
            (
                indicate_small * Qn
                + indicate_big * Qp
                + indicate_middle * (-q_w + q_w.round())
            )
            * grad_output
            * grad_scale
        )
        grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None