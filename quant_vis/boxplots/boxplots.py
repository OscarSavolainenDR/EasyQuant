import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from utils.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)


def per_channel_boxplots(
    weight_tensor: torch.Tensor, folder_path: Path, filename: str, title: str
):
    """
    Given a weight tensor, this function plots its per-output-channel boxplots.
    This can be used to observe the dynamic ranges of the channels.

    Inputs:
    - weight_tensor (torch.Tensor): the weight tensor whose output channels we are plotting boxplots for.
    - folder_path (pathlib.Path): the path to which we want to save the images. Can be None.
    - filename (str): the filename we will give to the image. Can be None.
    - title (str): the pre-fix title we will give to the image.

    If the folder_path or filename are None, then the plots will not be saved.

    """
    # Ensure the weight tensor is on the CPU
    weight_tensor = weight_tensor.cpu()

    # Get the number of output channels
    num_output_channels = weight_tensor.shape[0]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a list of data for each output channel
    data = [weight_tensor[i].flatten().tolist() for i in range(num_output_channels)]

    # Create the boxplots
    ax.boxplot(data)

    # Set the title and labels
    ax.set_title(f"{title} - Per-Output-Channel Weight Boxplots")
    ax.set_xlabel("Output Channel")
    ax.set_ylabel("Weight Value")

    # Set the x-tick labels
    ax.set_xticks(range(1, num_output_channels + 1))
    ax.set_xticklabels(range(num_output_channels))

    # Adjust the spacing and display the plot
    plt.tight_layout()
    plt.show()

    # Save file
    if not folder_path or not filename:
        logger.info("Boxplots are not being saved.")
        return
    file_path = os.path.join(folder_path, f"{filename}.png")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    fig.savefig(file_path, dpi=450)
