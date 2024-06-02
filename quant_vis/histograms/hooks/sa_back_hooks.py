import torch

from ...utils.act_data import ActData
from .forward_hooks import (
    add_activation_forward_hooks,
    activation_forward_histogram_hook,
)
from utils.dotdict import dotdict

from typing import Union, Callable
from utils.logger import setup_logger

# Configure the logger
logger = setup_logger(__name__)


def add_sensitivity_analysis_hooks(
    model: torch.nn.Module,
    conditions_met: Union[Callable, None] = None,
    bit_res: int = 8,
):
    """
    Adds the required forward and baxckwards hooks to gather the required data for the
    forward and backwards histograms, for the combined forward / sensitivity analysis
    plots.
    NOTE: the `bit_res` parameter does not control the quantization reoslution of the model, only of the
    histograms. Ideally they should match.

    `conditions_met` (Callable): This is a function that takes in a module and its name, and returns a boolean
                    indicating whether one should add the hook to it or not.
        Example:
            ```
            def conditions_met_forward_act_hook(module: torch.nn.Module, name: str) -> bool:
                if "hello" in name:
                    return True
                else:
                    return False
            ```
    """
    act_forward_histograms = add_activation_forward_hooks(
        model,
        conditions_met=conditions_met,
        hook=activation_forward_histogram_hook,
        bit_res=bit_res,
    )
    act_backward_histograms = add_sensitivity_backward_hooks(
        model, act_forward_histograms
    )

    return act_forward_histograms, act_backward_histograms


def add_sensitivity_backward_hooks(
    model: torch.nn.Module, act_forward_histograms: Union[None, ActData]
):
    """
    Adds the backwards hooks that gather the gradients and sums them up according to the forward
    histogram bins, so that one gets the summed gradients for each histogrma bin. If the output of the
    model is backpropagated without any manipulation of the loss, then these gradients will correspond
    to the relative contribution of each quantization bin to the output.
    """

    if act_forward_histograms is None:
        raise ValueError(
            "The `act_forward_histograms` parameter must be specified. It "
            "should be the output of `add_activation_forward_hooks`."
        )

    # We intialise a new ActData instance, which will be responsible for containing the
    # backwards pass data
    act_backward_histograms = ActData(data={}, hook_handles={})
    for module_name in act_forward_histograms.hook_handles.keys():
        module = model.get_submodule(module_name)
        hook_handle = module.register_full_backward_hook(
            backwards_SA_histogram_hook(
                act_forward_histograms, act_backward_histograms, module_name
            )
        )

        # We store the hook handles so that we can remove the hooks once we have finished
        # accumulating the backwards gradients.
        act_backward_histograms.hook_handles[module_name] = hook_handle

    return act_backward_histograms


def backwards_SA_histogram_hook(
    act_forward_histograms: ActData,
    act_backward_histograms: ActData,
    name: str,
):
    """
    A backward hook that measures the gradient being fed back through a quantization module.

    It requires that the `add_activation_forward_hooks` be called first.

    The hook will capture the backwards gradient, and map it to the same histogram bins as were
    used for the histograms in the forward hook. This will make it so that the gradients will be summed
    into bins that correspond to the forward values, so that they can be associated. If the output of the
    model is backpropagated without any manipulation of the loss, then these gradients will correspond
    to the relative contribution of each quantization bin to the output. I.e., it will correspond to a
    sensitivity analysis.

    backwards_histogram_hook inputs:
    - act_histogram (ActData): a dataclass instance that stores the activation histograms and hook handles.
    - name (str): the name of the module, and how its histogram will be stored in the dict.

    hook inputs:
    - module: the quantization module.
    - inp_grad: input gradient.
    - out_grad: output gradient.
    """

    def hook(module, inp_grad, out_grad):
        """
        Updates the dataclass `act_backward_histograms.data[name]` by summing up
        gradients computed with forward histograms and storing them as binned
        gradients in the corresponding bin indices.

        Args:
            module (ndarray.): 3D tensors that are passed through the forward call
                and then processed to compute the gradients with respect to the
                corresponding parameters.
                
                		- `name`: The name of the module.
                		- `data`: The dataclass instance containing the histogram data
                and gradients.
                		- `act_forward_histograms`: A dataclass instance containing the
                forward histogram bins and their corresponding indices.
                		- `act_backward_histograms`: A dataclass instance containing the
                backward histogram bins and their corresponding indices.
                		- `out_grad`: The gradient of the output with respect to the
                input, which is a tensor of shape `(1, 2)` containing the gradients
                of the two outputs.
            inp_grad (1D tensor.): 1D tensor of gradients to be summed and stored
                in the dataclass `act_backward_histograms`.
                
                		- `inp_grad` is a tensor with shape `(1, ..., 1)` containing the
                gradient of the forward pass for a single output dimension.
                		- The element at index `0` of `inp_grad` represents the gradient
                of the forward pass for the specified output dimension.
                		- The shape of `inp_grad` is determined by the
                `act_forward_histograms.data[name].hist.size()` and `bin_indices.flatten()`
                in the computation of `binned_grads`.
                		- `inp_grad.flatten()` returns a 1D tensor containing the flattened
                shape of the input gradient tensor.
                		- `torch.bincount(bin_indices.flatten(), weights=inp_grad.flatten())`
                computes the sum of gradients across all histogram bins, using the
                forward histogram bins as weights.
            out_grad (1D tensor.): 1D tensor of gradients output by the forward
                pass, which is used to compute the sum of histogram bins for each
                layer in the backward pass.
                
                		- `out_grad`: A tensor with shape `(1, 3)` containing the gradients
                of the model's outputs for the given name. The first dimension
                corresponds to the batch size, and the second and third dimensions
                correspond to the number of histogram bins in each output feature.

        """
        if name not in act_forward_histograms.data:
            return

        # Access the values-to-histogram-bins mapping from the forward call
        bin_indices = act_forward_histograms.data[name].bin_indices - 1
        grad = out_grad[0]

        # Compute the sum of gradients, with the forward histogram bins, using torch.bincount()
        size_diff = (
            act_forward_histograms.data[name].hist.size()[0] - bin_indices.max() - 1
        )
        padding = torch.zeros(size_diff)
        binned_grads = torch.concat(
            [torch.bincount(bin_indices.flatten(), weights=grad.flatten()), padding]
        )

        # Store the summed gradients into the dataclass
        back = dotdict()
        back.binned_grads = binned_grads
        act_backward_histograms.data[name] = back

    return hook
