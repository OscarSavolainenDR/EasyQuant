
import torch.nn as nn

def should_we_add_hooks(model: nn.Module) -> bool:
    """
    A function for checking if we should add the hooks to generate the visualization plots.
    One can customize this however one wishes.

    Inputs:
    - model: our quantizable model.

    Outputs:
    - return (bool): whether or not we should add the hooks and therefore generate the plots.
    """

    # Check if the model is a quantizable model, if not skip adding in activation hooks
    quantizable_model = False
    for name, _ in model.named_parameters():
        if "weight_fake_quant" in name:
            quantizable_model = True
    if not quantizable_model:
        return False
    
    # Once can add whatever conditions one wants to below.

    # insert conditions

    return True

