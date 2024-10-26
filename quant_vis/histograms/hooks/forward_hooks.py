import torch
import torch.quantization._numeric_suite as ns
from ...utils.act_data import ActData
from ...utils.hooks import is_model_quantizable
from utils.dotdict import dotdict

from ...settings import HIST_XMIN, HIST_XMAX, HIST_QUANT_BIN_RATIO

from typing import Callable, Union
from utils.logger import setup_logger

# Configure the logger
logger = setup_logger(__name__)


def activation_forward_histogram_hook(
    act_histogram: ActData, name: str, qscheme: torch.qscheme, bit_res: int = 8
):
    """
    A pre-forward hook that measures the floating-point activation being fed into a quantization module.
    This hook calculates a histogram, with the bins given by the quantization module's qparams,
    and stores the histogram in a global class.
    If the histogram for the given quantization module has not yet been initialised, this hook initialises
    it as an entry in a dict. If it has been initialised, this hook adds to it.

    Therefore, as more and more data is fed throuhg the quantization module and this hook,
    the histogram will accumulate the frequencies of all of the binned values.

    activation_histogram_hook inputs:
    - act_histogram (ActData): a dataclass instance that stores the activation histograms and hook handles.
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
        Generates high-quality documentation for code given to it, including a
        histogram of activation values for a specified quantization scheme.

        Args:
            module (ns.Shadow.): quant module for which the histogram is being
                computed, and it is used to determine the appropriate quantization
                range and histogram bins.
                
                		- `training`: Boolean indicating whether the module is being
                trained (True) or not (False).
                		- `type`: String indicating the type of the module (either
                "ns.Shadow" or the actual type of the module).
                		- `name`: String representing the name of the module.
                		- `quantization_bits`: Int representing the number of quantization
                bits used for the module's weights and activations.
                		- `zero_point`: Float representing the zero point for the module's
                quantization range.
                		- `scale`: Float representing the scale factor for the module's
                quantization range.
                		- `HIST_XMIN`: Int or Float representing the minimum value of
                the histogram bins.
                		- `HIST_XMAX`: Int or Float representing the maximum value of
                the histogram bins.
                		- `HIST_QUANT_BIN_RATIO`: Float representing the number of
                histogram bins per quantization bin.
                
                	The `module` object is then destructured to extract its properties
                and attributes, such as:
                
                		- `local_input`: The input tensor for the module.
                		- `qrange`: The quantization range for the module's weights and
                activations.
                		- `hist_min_bin`: The minimum bin index in the histogram.
                		- `hist_max_bin`: The maximum bin index in the histogram.
                		- `hist_bins`: An array of histogram bins, calculated based on
                the quantization range and the number of histogram bins per
                quantization bin.
                		- `tensor_histogram`: The resulting tensor histogram for the
                module's activations.
                		- `bin_indices`: An array of indices representing the histogram
                bins that the gradients should be mapped to.
            input (0D tensor of type `torch.Tensor`.): 1D tensor of activation
                values that will be processed through the histogramming operation.
                
                		- `type(module)` is not `ns.Shadow`: This indicates that the
                module is not a shadow module, and therefore the hook function
                should handle the quantization for this module.
                		- `name` in the `act_histogram` dict has not been initialized:
                This means that this is the first forward pass for this module,
                and the histogram entry for this module needs to be initialized.
                		- `local_input` is a tensor: This indicates that the input to
                the hook function is a tensor, which will be used to calculate the
                histogram bins.
                
                	The properties of the input tensor are not explained in this hook
                function, as they are not relevant to the quantization process.
                However, if necessary, these properties can be referenced in
                subsequent hook functions or in the main codebase.

        """
        if not module.training and type(module) is not ns.Shadow:

            # Get number of quantization bins from the quantization bit width
            qrange = 2**bit_res

            local_input = input[0].detach().cpu()

            # If the entry in the `act_histogram` dict has not been initialised, i.e. this is the first forward pass
            # for this module
            if name not in act_histogram.data:
                # We calculate the limits of the histogram. These are dependent on the qparams, as well as how
                # much "buffer" we want on either side of the quantization range, defined by `HIST_XMIN` and
                # `HIST_XMAX` and the qparams.
                hist_min_bin = (-HIST_XMIN * qrange - module.zero_point) * module.scale
                hist_max_bin = (
                    (HIST_XMAX + 1) * qrange - module.zero_point
                ) * module.scale

                # If symmetric quantization, we offset the range by half.
                if qscheme in (
                    torch.per_channel_symmetric,
                    torch.per_tensor_symmetric,
                ):
                    hist_min_bin -= qrange / 2 * module.scale
                    hist_max_bin -= qrange / 2 * module.scale

                # Create the histogram bins, with `HIST_QUANT_BIN_RATIO` histogram bins per quantization bin.
                hist_bins = (
                    torch.arange(
                        hist_min_bin.item(),
                        hist_max_bin.item(),
                        (module.scale / HIST_QUANT_BIN_RATIO).item(),
                    )
                    - (0.5 * module.scale / HIST_QUANT_BIN_RATIO).item()
                    # NOTE: offset by half a quant bin fraction, so that quantization centroids
                    # fall into the middle of a histogram bin.
                )
                # TODO: figure out a way to do this histogram on CUDA
                tensor_histogram = torch.histogram(local_input, bins=hist_bins)

                # Create a map between the histogram and values by using torch.bucketize()
                # The idea is to be able to map the gradients to the same histogram bins
                bin_indices = torch.bucketize(local_input, tensor_histogram.bin_edges)

                # Initialise stored histogram for this quant module
                stored_histogram = dotdict()
                stored_histogram.hist = tensor_histogram.hist
                stored_histogram.bin_edges = tensor_histogram.bin_edges
                stored_histogram.bin_indices = bin_indices

                # Store final dict in `act_histogram`
                act_histogram.data[name] = stored_histogram

            # This histogram entry for this quant module has already been intialised.
            else:
                # We use the stored histogram bins to bin the incoming activation, and add its
                # frequencies to the histogram.
                histogram = torch.histogram(
                    local_input,
                    bins=act_histogram.data[name].bin_edges.cpu(),
                )
                act_histogram.data[name].hist += histogram.hist

                # We overwrite the bin indices with the most recent bin indices
                bin_indices = torch.bucketize(local_input, histogram.bin_edges)
                act_histogram.data[name].bin_indices = bin_indices

    return hook


def add_activation_forward_hooks(
    model: torch.nn.Module,
    conditions_met: Union[Callable, None] = None,
    hook: Union[Callable, None] = None,
    bit_res: int = 8,
):
    """
    This function adds forward activation hooks to the quantization modules in
    the model, if their names match any of the patterns in
    `act_histogram.accepted_module_name_patterns`.
    These hooks measure and store values associated woth the activations,
    as defined by the hook. The attributes should be stored in `act_data.data`,
    and the hook handles should be stored in `act_data.hook_handles`.

    Inputs:
    - model (torch.nn.Module):      the model we will be adding hooks to.
    - conditions_met (Callable):    a function that returns True if the
                                    conditons are met for adding a hook to a
                                    module, and false otherwise. Defaults to
                                    None.
    - hook (Callable): the hook that will be added to the
                                    modules, which will store some activation
                                    attribute into `act_data`.
    - bit_res (int): the quantization bit width of the tensor, e.g. 8 for int8.

    Returns:
    - act_data (ActData): A dataclass instance that contains the
                                    stored histograms and hook handles.
    """

    # Check if the hook is specified
    if hook is None:
        raise ValueError("The `hook` argument must be specified.")

    # If the conditons are met for adding hooks
    if not is_model_quantizable(model, "activation"):
        logger.warning("None of the model activations are quantizable")
        return

    logger.warning(
        "\nAdding forward activation histogram hooks. This will significantly "
        "slow down the forward calls for the targetted modules."
    )

    # We intialise a new ActData instance, which will be responsible for
    # containing the per-channel max activation data
    act_data = ActData(data={}, hook_handles={})

    # Add activation-hist pre-forward hooks to the desired quantizable module
    for name, module in model.named_modules():
        if hasattr(module, "fake_quant_enabled") and "weight_fake_quant" not in name:
            if conditions_met and not conditions_met(module, name):
                logger.debug(
                    f"The conditons for adding an activation hook to module {name} were not met."
                )
                continue

            hook_handle = module.register_forward_pre_hook(
                hook(
                    act_data,
                    name,
                    module.qscheme,
                    bit_res=bit_res,
                )
            )
            # We store the hook handles so that we can remove the hooks once we have finished
            # accumulating the histograms.
            act_data.hook_handles[name] = hook_handle

    return act_data
