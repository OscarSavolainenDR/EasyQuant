import torch
import torch.quantization._numeric_suite as ns
from ..utils.global_vars import ActHistogram

from ..settings import HIST_XLIM_MIN, HIST_XLIM_MAX, HIST_BINS_PER_QUANT_BIN

import logging

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def activation_histogram_hook(act_histogram: ActHistogram, name: str, qscheme):
    """
    A pre-forward hook that measures the floating-point activation being fed into a quantization module.
    This hook calculates a histogram, with the bins given by the quantization module's qparams,
    and stores the histogram in a global class.
    If the histogram for the given quantization module has not yet been initialised, this hook initialises
    it as an entry in a dict. If it has been initialised, this hook adds to it.

    Therefore, as more and more data is fed throuhg the quantization module and this hook,
    the histogram will accumulate the frequencies of all of the binned values.

    activation_histogram_hook inputs:
    - act_histogram (ActHistogram): a dataclass instance that stores the activation histograms.
    - name (str): the name of the module, and how its histogram will be stored in the dict.
    - qscheme (XXXX): the qscheme of the quantization module.

    hook inputs:
    - module: the quantization module.
    - input: the activation fed to the quantization module.
    """

    def hook(module, input):
        # Ensure we are in eval mode, and ensure that this is not during a Shadow conversion check.
        if not module.training and type(module) is not ns.Shadow:

            # Get quantization resolution, i.e. 8 bits, 4 bits, etc.
            XXX
            qrange = 256

            # If the entry in the `act_histogram` dict has not been initialised, i.e. this is the first forward pass
            # for this module
            if name not in act_histogram.data:
                # We calculate the limits of the histogram. These are dependent on the qparams, as well as how
                # much "buffer" we want on either side of the quantization range, defined by `HIST_XLIM_MIN` and 
                # `HIST_XLIM_MAX`.

                # If symmetric quantization
                if qscheme in (
                    torch.per_channel_symmetric,
                    torch.per_tensor_symmetric,
                ):
                    hist_min_bin = (
                        -HIST_XLIM_MIN*qrange - module.zero_point - qrange/2
                    ) * module.scale
                    hist_max_bin = (
                        HIST_XLIM_MAX*qrange - module.zero_point - qrange/2
                    ) * module.scale

                # If affine quantization
                else:
                    hist_min_bin = (-HIST_XLIM_MIN*qrange - module.zero_point) * module.scale
                    hist_max_bin = (HIST_XLIM_MAX*qrange - module.zero_point) * module.scale

                # Create the histogram bins, with `HIST_BINS_PER_QUANT_BIN` histogram bins per quantization bin.
                hist_bins = (
                    torch.arange(
                        hist_min_bin.item(),
                        hist_max_bin.item(),
                        (module.scale / HIST_BINS_PER_QUANT_BIN).item(),
                    )
                    # - (0.5 * module.scale / HIST_BINS_PER_QUANT_BIN).item()
                )
                # NOTE: figure out a way to do this histogram on CUDA
                tensor_histogram = torch.histogram(
                    input[0].detach().cpu(), bins=hist_bins
                )

                # Initialise stored histogram for this quant module
                stored_histogram = {}
                stored_histogram['frequencies'] = tensor_histogram.hist.cuda()
                stored_histogram['bin_edges'] = tensor_histogram.bin_edges.cuda()

                # Store final dict in `act_histogram` 
                act_histogram.data[name] = stored_histogram

            # This histogram entry for this quant module has already been intialised.
            else:
                # We use the stored histogram bins to bin the incoming activation, and add its
                # frequencies to the histogram.
                histogram = torch.histogram(
                    input[0].detach().cpu(),
                    bins=act_histogram.data[name].bin_edges.cpu(),
                )

                act_histogram.data[name].hist += histogram.hist.cuda()

    return hook

def add_activation_hooks(model: torch.nn.Module):
    """
    This function adds activation hooks to the quantization modules in the model, if their names
    match any of the patterns in `act_histogram.accepted_module_name_patterns`.

    Inputs:
    - model (torch.nn.Module): the model we will be adding hooks to.
    """
    logger.warning(
        f"\nAdding activation histogram hooks. This will significantly slow down the forward calls for
        for the targetted modules."
    )

    # We sintialise a new ActHistogram instance, which will be responsible for containing the 
    # activation histogram data
    act_histograms = ActHistogram()

    # Add axtivation-hist pre-forward hooks to the desired quantizable module
    hooked_modules = []
    for name, module in model.named_modules():
        if hasattr(module, "activation_post_process") and hasattr(
            module.activation_post_process, "fake_quant_enabled"
        ):
            # We skip the modules whose names are not present in the `accepted_module_name_patterns`
            if not any(pattern in name for pattern in act_histograms.accepted_module_name_patterns):
                continue

            hooked_modules.append(name)
            hook_handle = module.activation_post_process.register_forward_pre_hook(
                activation_histogram_hook(
                    act_histograms,
                    name,
                    module.activation_post_process.qscheme,
                )
            )
            # We store the hook handles so that we can remove the hooks once we have finished
            # accumulating the histograms. 
            act_histograms.hook_handles[name] = hook_handle