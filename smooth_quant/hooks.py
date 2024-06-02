import torch
import torch.quantization._numeric_suite as ns
from .utils.act_data import ActData
from utils.dotdict import dotdict

from typing import Callable, Union
from utils.logger import setup_logger

# Configure the logger
logger = setup_logger(__name__)


def activation_forward_per_out_chan_max_hook(
    act_data: ActData, name: str, qscheme: torch.qscheme, bit_res: int = 8
):
    """
    A pre-forward hook that measures the floating-point activation being fed into a quantization module.
    This hook calculates the per-out-channel abs max value foe the given module.
    If the stiored max value for the given quantization module has not yet been initialised, this hook initialises
    it as an entry in a dict. If it has been initialised, this hook adds to it.

    Therefore, as more and more data is fed through the quantization module and this hook,
    the max value will update.

    activation_histogram_hook inputs:
    - act_data (ActData): a dataclass instance that stores the activation histograms and hook handles.
    - name (str): the name of the module, and how its histogram will be stored in the dict.
    - qscheme (torch.qscheme): the qscheme of the quantization module.
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    hook inputs:
    - module: the quantization module.
    - input: the activation fed to the quantization module.
    """

    def hook(module, input):
        # Ensure we are in eval mode, and ensure that this is not during a Shadow conversion check.
        if not module.training and type(module) is not ns.Shadow:

            local_input = input[0].detach().cpu()

            # If the entry in the `act_data` dict has not been initialised, i.e. this is the first forward pass
            # for this module
            if name not in act_data.data:

                # Store max val in `act_data`
                act_data.data[name] = input[0].abs().max(dim=1)
                XXX

            # This max mal entry for this quant module has already been intialised.
            else:
                # Update max val
                act_data.data[name] = torch.max(
                    act_data.data[name], input[0].abs().max(dim=1)
                )

    return hook
