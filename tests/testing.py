import torch
import torch.quantization as tq
import torch.ao.quantization as taq
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx
from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.qconfig_mapping import QConfigMapping
import torch.nn.quantized as nnq
import torch.fx as fx


from model.resnet import resnet18

from evaluate import evaluate

from ipdb_hook import ipdb_sys_excepthook

# import sys
# sys.path.append('BQ\quant_modules')
from quant_modules.state_toggling import enable_fake_quant, enable_PTQ_observer, disable_fake_quant, disable_PTQ_observer
from quant_modules.fake_quantize import BQFakeQuantize as FakeQuantize
from quant_modules.learnable_fake_quantize import BQLearnableFakeQuantize as LearnableFakeQuantize
from torch.ao.quantization.quantization_mappings import get_default_dynamic_quant_module_mappings
ipdb_sys_excepthook()


model = resnet18(weights='ResNet18_Weights.DEFAULT')
# print(model)

# Step 1: architecture changes
# QuantStubs (we will do FloatFunctionals later)
# Done

# Step 2: fuse modules (recommended but not necessary)
modules_to_list = model.modules_to_fuse()

# It will keep Batchnorm
model.eval()
# fused_model = torch.ao.quantization.fuse_modules_qat(model, modules_to_list)

# This will fuse BatchNorm weights into the preceding Conv
fused_model = torch.ao.quantization.fuse_modules(model, modules_to_list)

# Step 3: Assign qconfigs
backend = 'fbgemm'
# qconfig = torch.quantization.get_default_qconfig(backend)


fixed_act = lambda min , max : FixedQParamsFakeQuantize.with_args(
    observer=torch.ao.quantization.observer.FixedQParamsObserver.with_args(
        scale=(max - min) / 255.0,
        zero_point=-min / ((max - min) / 255.0),  #  zero_point = -min / scale
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    ),
)
learnable_weight = lambda channels : LearnableFakeQuantize.with_args(
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
#import torch.ao.nn.intrinsic.quantized.dynamic as nnqd
#mapping[torch.nn.Conv2d] = nnqd.Conv2d
fake_quant_model_dynamic = torch.quantization.quantize_dynamic(model, mapping=mapping)

# FX Mode
qconfig_FF = tq.QConfig(
    activation=learnable_fake_quant,
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_QS_DQ = tq.QConfig(
    activation=fixed_act(min=0, max=1),
    weight=tq.default_fused_per_channel_wt_fake_quant
)

qconfig_mapping = QConfigMapping().set_object_type((taq.QuantStub, taq.DeQuantStub), qconfig_QS_DQ) \
                                .set_object_type(nnq.FloatFunctional, qconfig_FF)

# Awkward we have to do this manually, just for the sake of accessing the `out_channels` attribute
for name, module in model.named_modules():
    if hasattr(module, 'out_channels'):
        qconfig = tq.QConfig(
            activation=learnable_fake_quant,
            weight=learnable_weight(channels=module.out_channels)
        )
        qconfig_mapping.set_module_name(name, qconfig)
        module.qconfig = qconfig


example_inputs = (torch.randn(1, 3, 224, 224),)

model.eval()
fx_model = prepare_qat_fx(model, qconfig_mapping, example_inputs)

# Evaluate
print('\noriginal')
#evaluate(model, 'cpu')
print('\nfused')
#evaluate(fused_model, 'cpu')
print('\ndynamic')
#evaluate(fake_quant_model_dynamic, 'cpu')
print('\n FX prepared')
fx_model.apply(enable_PTQ_observer)
#evaluate(fx_model, 'cpu')

# Step 5: convert (true int8 model)
fake_quant_model.to('cpu')
converted_model = torch.quantization.convert(fake_quant_model, inplace=False)

print('\nfake quant')
#evaluate(fake_quant_model, 'cpu')


print('\nconverted')
#evaluate(converted_model, 'cpu')

fake_quant_model.quant.activation_post_process.fake_quant_enabled = torch.Tensor([0])
# NOTE: these don't work! They don't trigger the setter at all!
fake_quant_model.apply(disable_fake_quant)
fake_quant_model.apply(disable_PTQ_observer)
import ipdb
ipdb.set_trace()
fake_quant_model.apply(disable_PTQ_observer)
#fake_quant_model.quant.activation_post_process.fake_quant_enabled[0] = 1
#
XXX

# NOTE: doesn't work for learnable for some reason.
fake_quant_model.apply(enable_fake_quant)
fake_quant_model.apply(enable_PTQ_observer)
# ## Torch compile
# compiled_model = torch.compile(model)
# print(compiled_model)