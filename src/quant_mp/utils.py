from copy import deepcopy

import torch
from tqdm import tqdm

from quant_mp.config import QuantModuleConfig
from quant_mp.QModules import QConv2d, QLinear

# Optional DeepSpeed import (ZeRO-3 sharding support during patching)
try:
    from deepspeed.zero import GatheredParameters as _DSGatheredParameters  # type: ignore

    class _MaybeGather:
        def __init__(self, params):
            self._params = [p for p in params if isinstance(p, torch.nn.Parameter)]

        def __enter__(self):
            self._ctx = _DSGatheredParameters(self._params, modifier_rank0=False)
            return self._ctx.__enter__()

        def __exit__(self, exc_type, exc, tb):
            return self._ctx.__exit__(exc_type, exc, tb)

except Exception:  # pragma: no cover - deepspeed not available

    class _MaybeGather:  # type: ignore
        def __init__(self, params):
            self._params = params

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False


# TODO: Maybe remove deprecated function
def replace_module(module, rconfig: QuantModuleConfig):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, torch.nn.Conv2d):
            bias = True if child_module.bias is not None else False
            new_module = QConv2d(
                child_module.in_channels,
                child_module.out_channels,
                child_module.kernel_size,  # pyright: ignore[reportArgumentType]
                child_module.stride,  # pyright: ignore[reportArgumentType]
                child_module.padding,  # pyright: ignore[reportArgumentType]
                child_module.dilation,  # pyright: ignore[reportArgumentType]
                child_module.groups,
                bias,
                child_module.padding_mode,
                rconfig,
            )
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, rconfig)

        if isinstance(child_module, torch.nn.Linear):
            bias = True if child_module.bias is not None else False
            new_module = QLinear(
                child_module.in_features,
                child_module.out_features,
                bias,
                rconfig,
            )
            setattr(module, child_name, new_module)
        else:
            replace_module(child_module, rconfig)


def patch_model(model, config: QuantModuleConfig):
    def replace_layer(module: torch.nn.Module):
        # Gather partitioned params under ZeRO-3 before reading state_dict
        params_local = list(module.parameters(recurse=False))
        # Use no-op context if deepspeed not present
        with _MaybeGather(params_local):
            if isinstance(module, torch.nn.Linear):
                bias = True if module.bias is not None else False
                new_module = QLinear(
                    module.in_features,
                    module.out_features,
                    bias,
                    module.weight.device,
                    module.weight.dtype,
                    config,
                )
                # Preserve original DS-managed parameters by rebinding
                # Avoids copying from potentially partitioned/empty tensors
                new_module.weight = module.weight
                if bias:
                    new_module.bias = module.bias

                return new_module
            elif isinstance(module, torch.nn.Conv2d):
                bias = True if module.bias is not None else False
                new_module = QConv2d(
                    module.in_channels,
                    module.out_channels,
                    module.kernel_size,  # pyright: ignore[reportArgumentType]
                    module.stride,  # pyright: ignore[reportArgumentType]
                    module.padding,  # pyright: ignore[reportArgumentType]
                    module.dilation,  # pyright: ignore[reportArgumentType]
                    module.groups,
                    bias,
                    module.padding_mode,
                    config,
                )
                # Rebind parameters to preserve DS sharded tensors
                new_module.weight = module.weight
                if bias:
                    new_module.bias = module.bias
                return new_module
            else:
                return module

    def recursive_setattr(obj: torch.nn.Module, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tqdm(tuple(model.named_modules()), desc="Patching Model"):
        if name and not name.endswith("lm_head"):
            recursive_setattr(model, name, replace_layer(module))
