import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
from typing import Tuple

from ...settings import HIST_QUANT_BIN_RATIO, HIST_XMAX, HIST_XMIN


############
# SUBPLOTS #
############
def fill_in_mean_subplot(
    distribution: torch.Tensor,
    zero_bin_value: torch.Tensor,
    clamped_prob_mass: torch.Tensor,
    ax_sub: matplotlib.axes._axes.Axes,
    color: str = "blue",
    data_name: str = "",
):
    """
    Fills in the summary sub-plot. This involves calculating the mean intra-bin values, and plotting them.
    We also add a few interesting statistics:
    - the amount of not-on-bin-centroid probability mass
    - the zero-bin value
    - the amount of clamped probability mass

    Inputs:
    - distribution (torch.Tensor): the PDF we will be getting the mean intra-bin plot of.
    - zero_bin_value (torch.Tensor): the zero bin probability mass value.
    - clamped_prob_mass (torch.Tensor): the clamped probability mass scalar value.
    - ax_sub (matplotlib.axes._axes.Axes) = the Axes object we will manipulate to fill in the subplot.
    - color (str): the color of the imean intra-bin plot.
    - data_name (str): part of the subtitle, e.g. "Forward Activation", "Gradient", etc.
    """
    # Sum every HIST_QUANT_BIN_RATIO'th value in the histogram.
    intra_bin = torch.zeros(HIST_QUANT_BIN_RATIO)
    for step in torch.arange(HIST_QUANT_BIN_RATIO):
        intra_bin[step] = distribution[step::HIST_QUANT_BIN_RATIO].sum()
    indices = range(HIST_QUANT_BIN_RATIO)
    intra_bin = intra_bin.numpy()

    # Plot the intra-bin behavior as subplot
    ax_sub.bar(indices, intra_bin, color=color)

    # Remove tick labels and set background to transparent for the overlay subplot
    ax_sub.set_xticks(np.arange(0, HIST_QUANT_BIN_RATIO + 1, HIST_QUANT_BIN_RATIO / 2))
    ax_sub.set_xticklabels(
        [
            f"{int(i)}/{HIST_QUANT_BIN_RATIO}"
            for i in np.arange(0, HIST_QUANT_BIN_RATIO + 1, HIST_QUANT_BIN_RATIO / 2)
        ]
    )
    ax_sub.set_xlim(-0.5, HIST_QUANT_BIN_RATIO + 0.5)
    ax_sub.patch.set_alpha(1)

    # Add title (with summary-ish statistics) and labels
    title_str = f"{data_name}\nMean Intra-bin Behavior\n(Not-on-quant-bin-centroid\nprob mass: {intra_bin[1:].sum():.2f})\nZero-bin mass: {zero_bin_value:.2f}"
    title_str += f"\nClamped prob mass: {clamped_prob_mass:.6f}"
    ax_sub.set_title(title_str)
    ax_sub.set_ylabel("Prob")
    ax_sub.set_xlabel("Bins (0 and 1 are centroids)")
    ax_sub.axvline(x=0, color="black", linewidth=1)
    ax_sub.axvline(x=HIST_QUANT_BIN_RATIO, color="black", linewidth=1)


def draw_centroids_and_tensor_range(
    ax: matplotlib.axes._axes.Axes,
    bin_edges: torch.Tensor,
    qrange: int,
    tensor_min_index: torch.Tensor,
    tensor_max_index: torch.Tensor,
    scale: torch.Tensor,
):
    """
    Draws black vertical lines at each quantization centroid, and adds thick red lines at the edges
    of the floating point tensor, i.e. highlights its dynamic range.

    Inputs:
    - ax (matplotlib.axes._axes.Axes): the Axes object we will be manipulating to add the plot elements.
    - bin_edges (torch.Tensor): the histogram bin edges
    - qrange (int): the number of quantization bins
    - tensor_min_index (torch.Tensor): the minimum value in the floating point tensor.
    - tensor_max_index (torch.Tensor): the maximum value in the floating point tensor.
    - scale (torch.Tensor): the quantization scale.
    """
    # Draws black vertical lines
    for index, x_val in enumerate(
        np.arange(
            start=bin_edges[int(HIST_XMIN * qrange * HIST_QUANT_BIN_RATIO)],
            stop=bin_edges[-int(HIST_XMAX * qrange * HIST_QUANT_BIN_RATIO)],
            step=scale,
        )
    ):
        if index == 0:
            ax.axvline(
                x=x_val,
                color="black",
                linewidth=0.08,
                label="Quantization bin centroids",
            )
        else:
            ax.axvline(x=x_val, color="black", linewidth=0.08)

    # Draw vertical lines at dynamic range boundaries of forward tensor (1 quantization bin padding)
    ax.axvline(
        x=bin_edges[tensor_min_index] - scale,
        color="red",
        linewidth=1,
        label="Tensor dynamic range",
    )
    ax.axvline(x=bin_edges[tensor_max_index] + scale, color="red", linewidth=1)


###################
# DATA PROCESSING #
###################
def get_weight_quant_histogram(
    weight: torch.nn.Parameter,
    scale: torch.nn.Parameter,
    zero_point: torch.nn.Parameter,
    qscheme: torch.qscheme,
    bit_res: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the histogram of the weight, with bins defined by its scale and zero-point.
    Unlike the activation, we plot the weight tensor on the integer scale. This is because:
    1) The weight tensor values are difficult to interpret anyway, so there isn't much to gain from the original scale.
    2) Normalizing by each channel's quantization parameters makes sense, so we can aggregate across channels.

    Inputs:
    - weight (torch.nn.Parameter): a weight tensor
    - scale (torch.nn.Parameter):  a qparam scale. This can be a single parameter, or in the case of per-channel quantization, a tensor with len > 1.
    - zero_point (torch.nn.Parameter): a qparam zero_point. This can be a single parameter, or in the case of per-channel quantization, a tensor with len > 1.
    - qscheme: specifies the quantization scheme of the weight tensor.
    - bit_res (int): the quantization bit width, e.g. 8 for 8-bit quantization.

    Outputs:
    - hist (Tuple[torch.Tensor, torch.Tensor]): torch.histogram output instance, with histogram and bin edges.
    """

    if qscheme in [torch.per_channel_affine, torch.per_channel_symmetric]:
        scale = scale.view(len(scale), 1, 1, 1)
        zero_point = zero_point.view(len(zero_point), 1, 1, 1)
    elif qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
        pass
    else:
        raise ValueError(
            "`qscheme` variable should be per-channel symmetric or affine, or per-tensor symmetric or affine"
        )

    # Weight tensor in fake-quantized space
    fake_quant_tensor = weight.detach() / scale.detach() + zero_point.detach()

    # Flatten the weight tensor
    fake_quant_tensor = fake_quant_tensor.reshape(-1)

    # Get number of quantization bins from the quantization bit width
    qrange = 2**bit_res

    # Calculate the histogram between `-HIST_XMIN * qrange` and `(1+HIST_MAX_XLIM) * qrange`, with `HIST_QUANT_BIN_RATIO` samples per quantization bin.
    # This covers space on either side of the 0-qrange quantization range, so we can see any overflow, i.e clamping.
    hist_bins = torch.arange(
        -HIST_XMIN * qrange,
        (1 + HIST_XMAX) * qrange,
        1 / HIST_QUANT_BIN_RATIO,
    )
    # If we are doing symmetric quantization, center the range at 0.
    if qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
        hist_bins -= qrange / 2

    hist = torch.histogram(fake_quant_tensor.cpu(), bins=hist_bins)
    return hist


def get_prob_mass_outside_quant_range(
    distribution: torch.Tensor, qrange: int
) -> torch.Tensor:
    """
    Returns the amount of probability mass outside the quantization range.
    """
    clamped_prob_mass = torch.sum(
        distribution[: int(HIST_XMIN * qrange * HIST_QUANT_BIN_RATIO)]
    ) + torch.sum(distribution[int((HIST_XMIN + 1) * qrange * HIST_QUANT_BIN_RATIO) :])
    return clamped_prob_mass


def moving_average(input_tensor, window_size):
    """
    Get a 1d moving average of a 1D torch tensor, used for creating a smoothed
    data distribution for the histograms.
    """
    # Create a 1D convolution kernel filled with ones
    kernel = torch.ones(1, 1, window_size) / window_size
    
    # Apply padding to handle boundary elements
    padding = (window_size - 1) // 2
    
    # Apply the convolution operation
    output_tensor = F.conv1d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    
    return output_tensor.squeeze()