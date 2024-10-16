from torchvision.models import resnet18
import torch
from quant_mp.QModules import QConv2d, QLinear

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

