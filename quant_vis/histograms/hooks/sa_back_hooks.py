import torch

from ...utils.act_histogram import ActHistogram
from .forward_hooks import add_activation_forward_hooks
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
        model, conditions_met=conditions_met, bit_res=bit_res
    )
    act_backward_histograms = add_sensitivity_backward_hooks(
        model, act_forward_histograms
    )

    return act_forward_histograms, act_backward_histograms


def add_sensitivity_backward_hooks(
    model: torch.nn.Module, act_forward_histograms: ActHistogram
):
    """
    Adds the backwards hooks that gather the gradients and sums them up according to the forward
    histogram bins, so that one gets the summed gradients for each histogrma bin. If the output of the
    model is backpropagated without any manipulation of the loss, then these gradients will correspond
    to the relative contribution of each quantization bin to the output.
    """

    # We intialise a new ActHistogram instance, which will be responsible for containing the
    # backwards pass data
    act_backward_histograms = ActHistogram(data={}, hook_handles={})
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
    act_forward_histograms: ActHistogram,
    act_backward_histograms: ActHistogram,
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
    - act_histogram (ActHistogram): a dataclass instance that stores the activation histograms and hook handles.
    - name (str): the name of the module, and how its histogram will be stored in the dict.

    hook inputs:
    - module: the quantization module.
    - inp_grad: input gradient.
    - out_grad: output gradient.
    """

    def hook(module, inp_grad, out_grad):
        """
        updates the backpropagation histogram dataclass with the gradients of a
        module, based on the forward histogram bins and using torch.bincount() to
        compute the sum of gradients.

        Args:
            module (instance of a class that defines the module.): 3D tensor of
                inputs to the hook function.
                
                		- `module`: This is an instance of a Python class that contains
                information about the model's forward pass and backward passes.
                		- `inp_grad`: This is a tensor representing the gradients of the
                model's inputs with respect to its outputs.
                		- `out_grad`: This is a tensor representing the gradients of the
                model's outputs with respect to its inputs.
                
                	The function then proceeds to compute and store the summed gradients
                of the histogram bins in the dataclass `act_backward_histograms`.
            inp_grad (ndarray.): 1-dimensional tensor of gradients to be summed
                with the forward histogram bins.
                
                		- `inp_grad` is an instance of the `torch.Tensor` class.
                		- It has one or more dimensions (depending on the input shape).
                		- The shape of the tensor is `(1,)` for a single-element tensor
                or a multi-dimensional tensor with dimensions `inp_grad.shape`.
                		- The elements of the tensor are floating-point values.
                		- The tensor may have a specific data type (e.g., `float32`,
                `float64`, etc.).
                		- The tensor is either a scalar or a vector, depending on the
                input shape.
            out_grad (1D tensor of size ( possibly zero).): 1D tensor of gradients
                to be summed and stored in the dataclass `act_backward_histograms`.
                
                		- `out_grad`: A tensor with shape `(1, 2)` containing the gradients
                for the current forward pass. The first dimension represents the
                batch size, and the second dimension represents the number of
                histogram bins.
                		- `inp_grad`: The input gradient tensor, which is a scalar
                representing the gradient of the current forward pass.

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
