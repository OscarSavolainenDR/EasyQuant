
###########################################
### Fusing Bn in ConvBnReLU into ConvReLU #
###########################################
import torch
from torch.ao.nn.intrinsic.qat.modules.conv_fused import (
    ConvBnReLU2d,
    ConvReLU2d,
    ConvBn2d,
)
from torch.ao.nn.qat import Conv2d
from typing import Union, Dict, Any, Tuple

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
    node: torch.fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module
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