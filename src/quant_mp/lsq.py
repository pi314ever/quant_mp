import torch
import math


def min_max_init(input, qconfig):
    if qconfig.qblock_size == "channel":
        xmax, _ = torch.max(torch.abs(input), dim=0)
    else:
        xmax = torch.max(torch.abs(input))
    if qconfig.qtype == "uniform":
        maxq = 2 ** (qconfig.qbits - 1) - 1
    elif qconfig.qtype == "float" and qconfig.format == "e2m1":
        maxq = 6
    elif qconfig.qtype == "float" and qconfig.format == "e3m0":
        maxq = 32
    else:
        raise NotImplementedError(f"Config not implemented for LSQ")

    scale = xmax / maxq
    return scale


def init_lsq(module):
    if module.rconfig.weight.alg == "lsq":
        if module.rconfig.weight.qblock_size == "channel":
            module.weight_clip_val = torch.nn.Parameter(
                torch.Tensor(module.input_features)
            )
        else:
            module.weight_clip_val = torch.nn.Parameter(torch.Tensor(1))
        scale = min_max_init(module.weight, module.rconfig.weight)

        module.weight_clip_val.data.copy_(scale)

        if module.rconfig.activation.alg == "lsq":
            module.activation_clip_val = torch.nn.Parameter(torch.tensor(float("nan")))


def init_lsq_activation(module, input):
    scale = min_max_init(input, module.rconfig.activation)
    module.activation_clip_val.data.copy_(scale)


class LsqBinaryTernaryExtension(torch.autograd.Function):
    """
    Modified from Learned Step-size Quantization.
    https://arxiv.org/abs/1902.08153
    """

    @staticmethod
    def forward(ctx, input, alpha, quantizer):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :return: quantized output
        """
        ctx.num_bits = quantizer.b

        Qn = torch.min(quantizer.G)
        Qp = torch.max(quantizer.G)

        eps = torch.tensor(0.00001, device=alpha.device).float()

        alpha = torch.where(alpha > eps, alpha, eps)

        grad_scale = (
            1.0 / math.sqrt(input.numel())
            if not Qp
            else 1.0 / math.sqrt(input.numel() * Qp)
        )
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp

        q_w, _ = quantizer.quant(input, (alpha, 0.0))

        w_q = q_w * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

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
        grad_alpha = torch.sum(grad_alpha, dim=0, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None
