import torch
import re
from .fake_quantize import BQFakeQuantize

#######################################
# QUANTIZATION STATE TOGGLING METHODS #
#######################################
def disable_fake_quant(mod):
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, BQFakeQuantize) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if isinstance(mod, BQFakeQuantize) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()

def disable_PTQ_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_PTQ_observer)

    """
    if isinstance(mod, BQFakeQuantize) or _is_fake_quant_script_module(mod):
        mod.disable_PTQ_observer()

def enable_PTQ_observer(mod):
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_PTQ_observer)

    """
    if isinstance(mod, BQFakeQuantize) or _is_fake_quant_script_module(mod):
        mod.enable_PTQ_observer()


def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        XXX
        # TODO: check this works, update paths
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub(r'\.___torch_mangle_\d+', '', suffix)
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or \
            name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False