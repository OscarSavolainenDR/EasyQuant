import torch
import os
import torch.quantization as tq
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx
from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.fx as fx

from pathlib import Path
from typing import Union, Dict, Any, Tuple

from model.resnet import resnet18
from tests.evaluate.evaluate import evaluate
from utils.ipdb_hook import ipdb_sys_excepthook

from quant_modules.state_toggling import (
    enable_fake_quant,
    enable_PTQ_observer,
    disable_fake_quant,
    disable_PTQ_observer,
)
from quant_modules.fake_quantize import EQFakeQuantize as FakeQuantize
from quant_modules.learnable_fake_quantize import (
    EQLearnableFakeQuantize as LearnableFakeQuantize,
)
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
)
from quant_vis.histograms import (
    plot_quant_act_hist,
    plot_quant_weight_hist,
    plot_quant_act_SA_hist,
    add_sensitivity_analysis_hooks,
    add_activation_forward_hooks,
)
from smooth_quant.hooks import activation_forward_per_out_chan_max_hook

ipdb_sys_excepthook()


model = resnet18(pretrained=True)
# print(model)

# Step 1: architecture changes
# QuantStubs (placed)
# Done

# Step 2: fuse modules (recommended but not necessary)
modules_to_list = model.modules_to_fuse()

# It will keep Batchnorm
model.eval()
# fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_list)

# This will fuse BatchNorm weights into the preceding Conv
fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)

# Step 3: Assign qconfigs
# backend = "fbgemm"
backend = "qnnpack"
# qconfig = torch.quantization.get_default_qconfig(backend)


fixed_act = lambda min, max: FixedQParamsFakeQuantize.with_args(
    observer=torch.ao.quantization.observer.FixedQParamsObserver.with_args(
        scale=(max - min) / 255.0,
        zero_point=-min / ((max - min) / 255.0),  #  zero_point = -min / scale
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
)
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

learnable_fake_quant = LearnableFakeQuantize.with_args(
    observer=tq.HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
    scale=1.0,
    zero_point=0.0,
    use_grad_scaling=True,
)

fake_quant = FakeQuantize.with_args(
    observer=tq.HistogramObserver,
    quant_min=0,
    quant_max=255,
    dtype=torch.quint8,
    qscheme=torch.per_tensor_affine,
)

qconfig = tq.QConfig(
    activation=learnable_fake_quant,
    weight=tq.default_fused_per_channel_wt_fake_quant,
)

torch.backends.quantized.engine = backend
from torch.ao.nn.intrinsic.modules.fused import ConvReLU2d

for name, module in fused_model.named_modules():
    module.qconfig = qconfig

fused_model.conv1.qconfig = tq.QConfig(
    activation=learnable_fake_quant,
    weight=learnable_weight(channels=64),
)
for module in fused_model.conv1.modules():
    module.qconfig = tq.QConfig(
        activation=learnable_fake_quant,
        weight=learnable_weight(channels=64),
    )

# Step 4: Prepare for fake-quant
fused_model.train()
fake_quant_model = torch.ao.quantization.prepare_qat(fused_model)
mapping = get_default_dynamic_quant_module_mappings()
fake_quant_model_dynamic = torch.quantization.quantize_dynamic(model, mapping=mapping)

qconfig_mapping = QConfigMapping()

# Awkward we have to do this manually, just for the sake of accessing the `out_channels` attribute
for name, module in model.named_modules():
    if hasattr(module, "out_channels"):
        qconfig = tq.QConfig(
            activation=learnable_fake_quant,
            weight=learnable_weight(channels=module.out_channels),
        )
        qconfig_mapping.set_module_name(name, qconfig)
        module.qconfig = qconfig


example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# Evaluate
print("\noriginal")
# evaluate(model, 'cpu')
print("\nfused")
# evaluate(fused_model, 'cpu')
print("\ndynamic")
# evaluate(fake_quant_model_dynamic, 'cpu')
print("\n FX prepared")
fx_model.apply(enable_PTQ_observer)
# evaluate(fx_model, 'cpu')

# Step 5: convert (true int8 model)
fake_quant_model.to("cpu")
converted_model = torch.quantization.convert(fake_quant_model, inplace=False)

print("\nfake quant")
# evaluate(fake_quant_model, 'cpu')


print("\nconverted")
# evaluate(converted_model, 'cpu')

fake_quant_model.quant.activation_post_process.fake_quant_enabled = torch.Tensor([0])
fake_quant_model.apply(disable_fake_quant)
fake_quant_model.apply(disable_PTQ_observer)
fake_quant_model.apply(disable_PTQ_observer)
# fake_quant_model.quant.activation_post_process.fake_quant_enabled[0] = 1
#

fake_quant_model.apply(enable_fake_quant)
fake_quant_model.apply(enable_PTQ_observer)
# ## Torch compile
# compiled_model = torch.compile(model)
# print(compiled_model)


###########################################
### Fusing Bn in ConvBnReLU into ConvReLU #
###########################################
from torch.ao.nn.intrinsic.qat.modules.conv_fused import (
    ConvBnReLU2d,
    ConvReLU2d,
    ConvBn2d,
)
from torch.ao.nn.qat import Conv2d


def fuse_conv_bn_relu_eval(
    conv: Union[ConvBnReLU2d, ConvBn2d]
) -> Union[ConvReLU2d, Conv2d]:
    """
    Given a quantizable ConvBnReLU2d Module returns a quantizable ConvReLU2d
    module such that the BatchNorm has been fused into the Conv, in inference mode.
    Given a ConvBn2d, it does the same to produce a Conv2d.
    One could also use `torch.nn.utils.fuse_conv_bn_eval` to produce a Conv, and then quantize that as desired.
    """
    assert not (conv.training or conv.bn.training), "Fusion only for eval!"
    qconfig = conv.qconfig
    if type(conv) is ConvBnReLU2d:
        new_conv = ConvReLU2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            qconfig=qconfig,
        )
    elif type(conv) is ConvBn2d:
        new_conv = Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
            conv.padding_mode,
            qconfig=qconfig,
        )

    new_conv.weight, new_conv.bias = fuse_conv_bn_weights(
        conv.weight,
        conv.bias,
        conv.bn.running_mean,
        conv.bn.running_var,
        conv.bn.eps,
        conv.bn.weight,
        conv.bn.bias,
    )

    return new_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    """
    Helper function for fusing a Conv and BatchNorm into a single weight/bias tensor pair.
    """
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape(
        [-1] + [1] * (len(conv_w.shape) - 1)
    )
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


# Graph manipulation functions for fusing Convs and BatchNorms
def _parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


def replace_node_module(
    node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
):
    """
    Helper function for having `new_module` take the place of `node` in a dict of modules.
    """
    assert isinstance(node.target, str)
    parent_name, name = _parent_name(node.target)
    # modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def convbn_to_conv(fx_model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Iterates through the graph nodes, and:
    - where it finds a ConvBnReLU it replaces it with ConvReLU
    - where it finds a ConvBn it replaces it with Conv

    This function works in-place on `fx_model`.

    Inputs:
    fx_model (torch.fx.GraphModule): a graph module, that we want to perform transformations on

    Output:
    (torch.fx.GraphModule): a model where we have swapped out the 2d ConvBn/ConvBnReLU for Conv/ConvReLU, and
                            fused the Bns into the Convs.
    """
    modules = dict(fx_model.named_modules())

    for node in fx_model.graph.nodes:
        # If the operation the node is doing is to call a module
        if node.op == "call_module":
            # The current node
            orig = fx_model.get_submodule(node.target)
            if type(orig) in [ConvBnReLU2d, ConvBn2d]:
                # Produce a fused Bn equivalent.
                fused_conv = fuse_conv_bn_relu_eval(orig)
                # This updates `modules` so that `fused_conv` takes the place of what was represented by `node`
                replace_node_module(node, modules, fused_conv)

    return fx_model


fx_model.eval()
fx_model: torch.fx.GraphModule = convbn_to_conv(fx_model)
input = example_inputs[0]
out = fx_model(input)  # Test we can feed something through the model


def conditions_met_forward_act_hook(module: torch.nn.Module, name: str) -> bool:
    """
    Evaluates whether a given `name` meets certain conditions by returning `True`
    if any of the conditions are met, or `False` otherwise.

    Args:
        module (torch.nn.Module): module where the function is defined.
        name (str): boolean value of whether the input string is equal to the
            literal value "1".

    Returns:
        bool: a Boolean value indicating whether the condition specified in the
        `if` statement is true or false.

    """
    if "1" in name:
        return True
    return False


def weight_conditions_met(module: torch.nn.Module, name: str) -> bool:
    """
    Evaluates if a given `name` has the substring `"conv"`. If it does, the function
    returns `True`, otherwise it returns `False`.

    Args:
        module (torch.nn.Module): Python module whose class definition is being
            checked for existence.
        name (str): string to be checked for existence of the keyword `conv` in it.

    Returns:
        bool: `True` if the given name satisfies the condition, and `False` otherwise.

    """
    if "conv" not in name:
        return False
    return True


# act_forward_histograms = add_activation_forward_hooks(
# fx_model, conditions_met_forward_act_hook
# )
fx_model.apply(enable_fake_quant)
fx_model.apply(enable_PTQ_observer)

evaluate(fx_model, "cpu")
fx_model.apply(disable_PTQ_observer)

from quant_vis.boxplots import per_channel_boxplots

per_channel_boxplots(
    fx_model.conv1.weight,
    folder_path=Path(os.path.abspath("") + "/Box_plots"),
    filename="conv1",
    title="conv1",
)

act_forward_max_val = add_activation_forward_hooks(
    model,
    conditions_met=conditions_met_forward_act_hook,
    activation_forward_hook=activation_forward_per_out_chan_max_hook,
    bit_res=8,
)

# sum_pos_1 = [0.18, 0.60, 0.1, 0.1]
# sum_pos_2 = [0.75, 0.43, 0.1, 0.1]
# plot_quant_weight_hist(
#     fx_model,
#     plot_title="TEST",
#     file_path = Path(__file__).resolve().parent / "Histogram_plots",
#     module_name_mapping=None,
#     sum_pos_1=sum_pos_1,
#     conditions_met=weight_conditions_met,
#     bit_res=8,
# )
# plot_quant_act_hist(
# act_forward_histograms,
# plot_title="TEST",
# file_path = Path(__file__).resolve().parent / "Histogram_plots",
# module_name_mapping=None,
# sum_pos_1=sum_pos_1,
# bit_res=8,
# )

# MANUALLY SET 1ST QUANTSTUB TO BE SCALE 1/255
# fx_model.activation_post_process_0.scale.data = torch.Tensor([1/255])
# fx_model.activation_post_process_0.zero_point.data = torch.Tensor([0.0])


# act_forward_histograms, act_backward_histograms = add_sensitivity_analysis_hooks(
#     fx_model, conditions_met_forward_act_hook
# )
# for _ in range(2):
#     # input = torch.rand(1,3,256,256)
#
#     import urllib
#
#     url, filename = (
#         "https://github.com/pytorch/hub/raw/master/images/dog.jpg",
#         "dog.jpg",
#     )
#     try:
#         urllib.URLopener().retrieve(url, filename)
#     except:
#         urllib.request.urlretrieve(url, filename)
#     # sample execution (requires torchvision)
#     from PIL import Image
#     from torchvision import transforms
#
#     input_image = Image.open(filename)
#     preprocess = transforms.Compose(
#         [
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]
#     )
#     input_tensor = preprocess(input_image)
#     input_batch = input_tensor.unsqueeze(
#         0
#     )  # create a mini-batch as expected by the model
#     # JUST FOR TESTING THE PERFECT QUANT OF FIRST QUANTSTUB
#     # input_batch = torch.randint(0, 255, (1, 3, 256, 256))/ 255
#     output = fx_model(input_batch)
#
#     output.mean().backward()
#
# plot_quant_act_SA_hist(
#     act_forward_histograms,
#     act_backward_histograms,
#     file_path = Path(__file__).resolve().parent / "Histogram_plots",
#     sum_pos_1=sum_pos_1,
#     sum_pos_2=sum_pos_2,
#     plot_title="TEST-SA",
#     module_name_mapping=None,
#     bit_res=8,
# )
# XXX
#
# TODO: test conditions met callable

