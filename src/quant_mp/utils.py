
import torch
from quant_mp.QModules import QConv2d, QLinear

from copy import deepcopy
import torch
from quant_mp.QModules import QLinear, init_lsq
from quant_mp.config import rconfig


def replace_module(module, rconfig: rconfig):
    for child_name, child_module in module.named_children():

        if isinstance(child_module, torch.nn.Conv2d):
            new_module = QConv2d(rconfig,
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
            replace_module(child_module, rconfig)


        if isinstance(child_module, torch.nn.Linear):
            new_module = QLinear(
                        child_module.in_features, 
                        child_module.out_features,
                        rconfig,
            )
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, rconfig)



def patch_model(model, config: rconfig):

    def replace_layer(module: torch.nn.Module):
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

    def recursive_setattr(obj: torch.nn.Module, attr, value):
        attr = attr.split('.', 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tuple(model.named_modules()):
        if name and not name.endswith('lm_head'):
            recursive_setattr(model, name, replace_layer(module))

