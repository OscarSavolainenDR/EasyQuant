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
        """
        Updates the maximum value stored in the `act_data` dictionary for a given
        module based on the input to that module.

        Args:
            module (`ns.Shadow`.): neural network module for which the maximum
                value is being calculated in the function.
                
                		- `module`: A PyTorch module object that represents the current
                module being processed in the forward pass.
                		- `training`: A boolean variable that indicates whether the
                module is in training or inference mode. If `training` is `True`,
                the module is in training mode, and if `training` is `False`, the
                module is in inference mode.
                		- `type`: A string variable that represents the type of the
                module. In this case, `type` is either `ns.Shadow` or `module`.
                		- `local_input`: A PyTorch tensor object that represents the
                input to the current module. `local_input` is created by detaching
                and converting the input tensor from the `act_data` dictionary
                into a CPU tensor.
                		- `name`: A string variable that represents the name of the
                current module. This variable is used to store the max value in
                the `act_data` dictionary for each quant module.
                
                	The logic of the `hook` function can be broken down into two branches:
                
                	1/ If the entry in the `act_data` dict has not been initialized,
                i.e., this is the first forward pass for this module, then:
                			- Store the max value in `act_data` using the input tensor `local_input`.
                			- Set the value of `name` to the name of the current module.
                	2/ Otherwise, if the max value for this quant module has already
                been initialized, then:
                			- Update the max value in `act_data` using the input tensor `local_input`.
                			- The updated value is stored in `act_data.data[name]`.
            input (0-dimensional tensor (scalar).): 0-dimensional tensor that
                contains the input data for the module.
                
                		- `type(module)`: The type of the module that triggered the hook.
                In this case, it is `ns.Shadow`.
                		- `module.training`: A boolean indicating whether the module is
                in training mode or not. In this case, it is `False`.
                		- `input[0].detach().cpu()`: The first input to the module,
                detached from its original tensor and converted to a CPU tensor
                for further processing.
                		- `name`: The name of the module.
                		- `act_data`: A dictionary containing data related to the module's
                activation function. In this case, it is used to store the maximum
                value found in the input tensor for each module.
                
                	The code within the `if` block explains how to handle the
                initialisation of the `act_data` dict and how to update its values
                based on the input tensor.

        """
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
