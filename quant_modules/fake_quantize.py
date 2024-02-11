"""
File for the fake-quant modules. This is a DIRECT analog to PyTorch's
`torch.ao.quantization.fake_quantize.py` `FakeQuantize` class.

There are two main differences to PyTorch's `FakeQuantize` classL
Firstly, the forward call is split into modular parts:
    - an observation forward call, and
    - a transform forward call.
This allows one to experiment with different observer algorithms and
quantization transforms.

Secondly, this also speeds up the execution significantly, because the branching
if-statement logic is front-loaded to when one chooses which behavior
one wants for the quantization module, and doesn't require that one answer
branching if-statements on the fly in every forward call.

"""

import torch
from torch.ao.quantization.fake_quantize import FakeQuantize as TorchFakeQuantize
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    HistogramObserver,
    MovingAveragePerChannelMinMaxObserver,
    FixedQParamsObserver,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
)

__all__ = [
    "BQFakeQuantize",
    "BQFixedQParamsFakeQuantize",
    "disable_fake_quant",
    "disable_PTQ_observer",
    "enable_fake_quant",
    "enable_PTQ_observer",
    "default_fake_quant",
    "default_weight_fake_quant",
    "default_dynamic_fake_quant",
    "default_fixed_qparams_range_neg1to1_fake_quant",
    "default_fixed_qparams_range_0to1_fake_quant",
    "default_symmetric_fixed_qparams_fake_quant",
    "default_affine_fixed_qparams_fake_quant",
    "default_per_channel_weight_fake_quant",
    "default_embedding_fake_quant",
    "default_embedding_fake_quant_4bit",
    "default_histogram_fake_quant",
]

# Functions for sorting the qscheme
def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_symmetric, torch.per_channel_affine, torch.per_channel_affine_float_qparams]

def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]

def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]

def _is_float_qparams(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_channel_affine_float_qparams, ]


class BQFakeQuantize(TorchFakeQuantize):
    r"""This is a wrapper around Torch's `FakeQuantize` class. It modularizes the forward calls,
    speeding up their execution and allowing for the scalable introduction of different `observer`
    and `transform` algorithms.

    `observer_enabled` is renamed to the more descriptive `PTQ_observer_enabled`, also
    enabling the introduction of observers that are not PTQ.

    The qparams of this quantization module can only be updated manually, or via observation.

    The output of this module is given by::
        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    """
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, is_dynamic=False, **observer_kwargs):
        super().__init__(observer=observer, quant_min=quant_min, quant_max=quant_max, is_dynamic=is_dynamic, **observer_kwargs)
        delattr(self, 'observer_enabled')
        self.register_buffer('PTQ_observer_enabled', torch.tensor([0], dtype=torch.uint8))
        self.observer = self._get_PTQ_forward()
        self.transform = self._get_fake_quant_forward()


    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def forward(self, X):
        """
        The forward call of the quantization modules is a composition:
        one does observation, and then one returns a tensor of some sort.

        Depending on what the `observer` and `transform` forward calls are equal to,
        one can perform any of the combinations of operations that one may want, e.g.:
        - PTQ and return a floating point tensor;
        - no observation at all and return a fake-quantized tensor;
        """
        X = self.observer(X)
        X = self.transform(X)
        return X

    ##########################
    # FORWARD METHOD SETTERS #
    ##########################
    def enable_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 1
        self.transform = self._get_fake_quant_forward()

    def disable_fake_quant(self) -> None:
        self.fake_quant_enabled[0] = 0
        self.transform = self._get_dummy_forward()

    def enable_PTQ_observer(self) -> None:
        self.PTQ_observer_enabled[0] = 1
        self.observer = self._get_PTQ_forward()

    def disable_PTQ_observer(self) -> None:
        self.PTQ_observer_enabled[0] = 0
        self.observer = self._get_dummy_forward()


    ##########################
    # FORWARD METHOD GETTERS #
    ##########################
    def _get_dummy_forward(self):
        """
        Returns the dummy forward call.
        """
        return self.dummy_forward

    def _get_fake_quant_forward(self):
        """
        Returns the fake-quantize forward call, depending on the qscheme.
        """
        if _is_per_channel(self.qscheme):
            return self.fake_quant_per_channel_forward
        elif _is_per_tensor(self.qscheme):
            return self.fake_quant_per_tensor_forward
        else:
            raise NotImplementedError("FakeQuantize only supports per-channel or per-tensor quantization")

    def _get_PTQ_forward(self):
        """
        Returns the PTQ forward call.
        """
        return self.PTQ_forward

    #################
    # FORWARD CALLS #
    #################
    def dummy_forward(self, X):
        """
        Returns the input tensor. This can be interpreted as short-circuiting
        the observer in the case of the `observer` forward call, and returning
        a dequantized tensor in the case of the `transform` forward call.
        """
        return X

    ###########################
    # TRANSFORM FORWARD CALLS #
    ###########################
    def fake_quant_per_channel_forward(self, X):
        """
        This is the forward call that returns a fake-quantized tensor. It
        does per-channel quantization.
        """
        X = torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point,
                    self.ch_axis, self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    def fake_quant_per_tensor_forward(self, X):
        """
        This is the forward call that returns a fake-quantized tensor. It
        does per-tensor quantization.
        """
        X = torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.activation_post_process.quant_min, self.activation_post_process.quant_max)
        return X

    ##########################
    # OBSERVER FORWARD CALLS #
    ##########################
    def PTQ_forward(self, X):
        """
        This forward call performs PTQ observation on the tensor, updating
        the qparams of `self`, i.e. the quantization module.
        """
        # Calls PTQ observer
        self.activation_post_process(X.detach())
        # Returns the qparams calculated by the PTQ observer
        _scale, _zero_point = self.calculate_qparams()
        # Overwrites the scale and zero-point of `self`
        _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
        if self.scale.shape != _scale.shape:
            self.scale.resize_(_scale.shape)
            self.zero_point.resize_(_zero_point.shape)
        self.scale.copy_(_scale)
        self.zero_point.copy_(_zero_point)
        return X

    # TODO: add GPTQ and AWQ

    ############
    # PRINTING #
    ############
    @torch.jit.export
    def extra_repr(self):
        return 'observer_call={}, transform_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.observer.__name__, self.transform.__name__, self.fake_quant_enabled, self.PTQ_observer_enabled,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)


class BQFixedQParamsFakeQuantize(BQFakeQuantize):
    """Simulate quantize and dequantize in training time.

    Simulate quantize and dequantize with fixed quantization
    parameters in training time. Only per tensor quantization
    is supported.
    """

    # TODO: rename observer to observer_ctr
    def __init__(self, observer):
        super().__init__(observer=observer)
        assert type(self.activation_post_process) == FixedQParamsObserver, \
            f"{self.__class__.__name__}'s observer must be a {FixedQParamsObserver.__name__}"
        self._observer_ctr = observer
        self.scale = self.activation_post_process.scale
        self.zero_point = self.activation_post_process.zero_point
        assert _is_per_tensor(self.qscheme), 'Only per tensor quantization is supported' + \
            ' FixedQParamsFakeQuantize module, got qscheme:' + str(self.qscheme)

    @torch.jit.export
    def calculate_qparams(self):
        return self.scale, self.zero_point

    @torch.jit.export
    def extra_repr(self):
        """Define a string representation of the object's attributes."""
        return 'observer_call={}, transform_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, scale={}, zero_point={}, ' \
               'dtype={}, quant_min={}, quant_max={}, qscheme={}'.format(
                   self.observer.__name__, self.transform.__name, self.fake_quant_enabled, self.PTQ_observer_enabled,
                   self.scale, self.zero_point, self.dtype,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.qscheme)



default_fake_quant = BQFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
"""
Default fake_quant for activations.
"""

default_weight_fake_quant = BQFakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
"""
Default fake_quant for weights.
Observer is memoryless since averaging_constant is 1.
"""

default_dynamic_fake_quant = BQFakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, is_dynamic=True,
    dtype=torch.quint8, averaging_constant=1)
"""
Default dynamic fake_quant for activations.
"""

default_fixed_qparams_range_neg1to1_fake_quant = (
    BQFixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_neg1to1_observer)
)
default_fixed_qparams_range_0to1_fake_quant = (
    BQFixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_0to1_observer)
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_fake_quant = default_fixed_qparams_range_neg1to1_fake_quant
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

default_per_channel_weight_fake_quant = BQFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                               quant_min=-128,
                                                               quant_max=127,
                                                               dtype=torch.qint8,
                                                               qscheme=torch.per_channel_symmetric,
                                                               reduce_range=False,
                                                               ch_axis=0)
"""
Default fake_quant for per-channel weights.
Observer is memoryless since averaging_constant is 1.
"""
default_embedding_fake_quant = BQFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                      qscheme=torch.per_channel_affine_float_qparams,
                                                      dtype=torch.quint8,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      ch_axis=0,
                                                      averaging_constant=1)
"""
Default fake_quant for embeddings.
Observer is memoryless since averaging_constant is 1.
"""

default_embedding_fake_quant_4bit = BQFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                           qscheme=torch.per_channel_affine_float_qparams,
                                                           ch_axis=0,
                                                           dtype=torch.quint4x2,
                                                           averaging_constant=1)

"""
Fake_quant for activations using a histogram..
"""
default_histogram_fake_quant = BQFakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      reduce_range=True)