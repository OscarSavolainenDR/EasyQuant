import torch
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping

from pathlib import Path
import ipdb

from ...tests.model.resnet import resnet18
from ...tests.evaluate import evaluate

from .fusing_bn_into_conv import convbn_to_conv
from quant_modules.state_toggling import (
    enable_fake_quant,
    enable_PTQ_observer,
    disable_fake_quant,
    disable_PTQ_observer,
)
from quant_modules.learnable_fake_quantize import (
    BQLearnableFakeQuantize as LearnableFakeQuantize,
)
from quant_vis.histograms import (
    plot_quant_act_hist,
    plot_quant_weight_hist,
    plot_quant_act_SA_hist,
    add_sensitivity_analysis_hooks,
    add_activation_forward_hooks,
)

#######################
# INITIALISE FX MODEL #
#######################
# Intialise model
model = resnet18(weights="ResNet18_Weights.DEFAULT")

# Initialise qconfigs
learnable_weight = lambda channels: LearnableFakeQuantize.with_args(
    observer=tq.PerChannelMinMaxObserver,
    quant_min=-128,
    quant_max=127,
    dtype=torch.qint8,
    qscheme=torch.per_channel_symmetric,
    scale=0.1,
    zero_point=0.0,
    use_grad_scaling=True,
    channel_len=channels,
)

learnable_act = LearnableFakeQuantize.with_args(
    observer=tq.HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    scale=1.0,
    zero_point=0.0,
    use_grad_scaling=True,
)

# Assign qconfigs
torch.backends.quantized.engine = "fbgemm"
qconfig_mapping = QConfigMapping()
for name, module in model.named_modules():
    if hasattr(module, "out_channels"):
        qconfig = tq.QConfig(
            activation=learnable_act,
            weight=learnable_weight(channels=module.out_channels),
        )
        qconfig_mapping.set_module_name(name, qconfig)
        module.qconfig = qconfig


# Create the FX quantized model
example_inputs = (torch.randn(1, 3, 224, 224),)
model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)


############
# EVALUATE #
############
print("\n FX prepared")
fx_model.apply(enable_PTQ_observer)
evaluate(fx_model, 'cpu')

# Make sure we can convert the model correctly
fx_model.to("cpu")
converted_model = torch.quantization.convert(fx_model, inplace=False)

print("\nconverted")
evaluate(converted_model, 'cpu')

##############################
# FUSE BATCHNORMS INTO CONVS #
##############################
# Fuse BatchNorms into preceding Conv layers. This makes plotting the
# distribution of weight tensors more interpretable.
fx_model.eval()
fx_model: torch.fx.GraphModule = convbn_to_conv(fx_model)

def conditions_met_forward_act_hook(module):
    ipdb.set_trace()
    return True

# We run data through the model so we can measure the activations for the plots
fx_model.apply(enable_fake_quant)
fx_model.apply(enable_PTQ_observer)
evaluate(fx_model, "cpu")
fx_model.apply(disable_PTQ_observer)

act_forward_histograms, act_backward_histograms = add_sensitivity_analysis_hooks(
    fx_model, conditions_met_forward_act_hook
)

#########################################
# FEEDING DATA THROUGH THE MODEL FOR SA #
#########################################
import urllib
url, filename = (
    "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
    "dog.jpg",
)
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)
# sample execution (requires torchvision)
from PIL import Image
from torchvision import transforms

input_image = Image.open(filename)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(
    0
)  # create a mini-batch as expected by the model

# Feed data through the model
output = fx_model(input_batch)

# We backpropagate the gradients. We take the mean of the output, ensuring that
# we backprop a scalar where all outputs are equally represented.
output.mean().backward()

##############################
# SENSITIVITY ANALYSIS PLOTS #
##############################
# Generate the forward and Sensitivity Analysis plots
plot_quant_act_SA_hist(
    act_forward_histograms,
    act_backward_histograms,
    file_path=Path("Histogram_plots"),
    sum_pos_1=[0.18, 0.60, 0.1, 0.1], # location of the first mean intra-bin plot
    sum_pos_2=[0.75, 0.43, 0.1, 0.1],
    plot_title="TEST-SA",
    module_name_mapping=None,
    bit_res=8,  # This should match the quantization resolution. Changing this will not change the model quantization, only the plots.
)


# Produce only the forward pass histogram plots
act_forward_histograms = add_activation_forward_hooks(
    fx_model, conditions_met_forward_act_hook
)
evaluate(fx_model, 'cpu')
plot_quant_act_hist(
    act_forward_histograms,
    file_path=Path("Histogram_plots"),
    plot_title='Just forward hists',
    module_name_mapping=None,
)

# Produce weight histogram plots (no data required)
plot_quant_weight_hist(
    fx_model,
    file_path=Path("Histogram_plots"),
    plot_title='Just weight hists',
    module_name_mapping=None,
)