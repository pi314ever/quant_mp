from torchvision.models import resnet18
import torch
from quant_mp.QModules import QConv2d, QLinear

from copy import deepcopy
from transformers import AutoModelForCausalLM
import torch
from quant_mp.QModules import QLinear
from quant_mp.config import rconfig, qconfig
from quant_mp.lsq import init_lsq

def replace_module(module, qconfig):
    for child_name, child_module in module.named_children():

        if isinstance(child_module, torch.nn.Conv2d):
            new_module = QConv2d(qconfig,
                        child_module.in_channels, 
                        child_module.out_channels,
                        child_module.kernel_size,
                        child_module.stride,
                        child_module.padding,
                        child_module.dilation,
                        child_module.groups,
                        child_module.bias,
                        child_module.padding_mode)
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, qconfig)


        if isinstance(child_module, torch.nn.Linear):
            new_module = QLinear(
                        child_module.in_features, 
                        child_module.out_features,
                        qconfig,
            )
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, qconfig)



def patch_model(model, config):

    def replace_layer(module):
        if isinstance(module, torch.nn.Linear):
            target_state_dict   = deepcopy(module.state_dict())
            bias                = True if module.bias is not None else False
            new_module          = QLinear(
                                    module.in_features,
                                    module.out_features,
                                    config,
                                    bias,
                                )
            new_module.load_state_dict(target_state_dict, strict=False)
            if config.weight.alg == 'lsq':
                init_lsq(new_module)

            return new_module
        else:
            return module

    def recursive_setattr(obj, attr, value):
        attr = attr.split('.', 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tuple(model.named_modules()):
        if name:
            recursive_setattr(model, name, replace_layer(module))

