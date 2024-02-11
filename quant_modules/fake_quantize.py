"""
File for the fake-quant modules. This is a DIRECT analog to PyTorch's
`torch.ao.quantization.fake_quantize.py`.

The main difference is that we break up the forward calls into minimal blocks, and use
setter methods to swap out the forward calls of the quantization modules to the
minimal needed. This significantly increases the speed of forward and backwards
passes, admittedly at the cost of some added coding complexity.

However, this makes understanding the state of the quantization modules easier,
as there is no ambiguity about what is happening inside any specific forward call.
"""

import torch
from torch.nn import Module
from torch.ao.quantization.observer import (
    MovingAverageMinMaxObserver,
    HistogramObserver,
    MovingAveragePerChannelMinMaxObserver,
    FixedQParamsObserver,
    default_fixed_qparams_range_0to1_observer,
    default_fixed_qparams_range_neg1to1_observer,
    _with_args,
)
import re
from abc import ABC, abstractmethod
from typing import Any, Tuple

__all__ = [
    "FakeQuantizeBase",
    "FakeQuantize",
    "FixedQParamsFakeQuantize",
    "FusedMovingAvgObsFakeQuantize",
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
    "default_fused_act_fake_quant",
    "default_fused_wt_fake_quant",
    "default_fused_per_channel_wt_fake_quant",
    "fused_wt_fake_quant_range_neg_127_to_127",
    "fused_per_channel_wt_fake_quant_range_neg_127_to_127",
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


class FakeQuantizeBase(ABC, Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    PTQ_observer_enabled: torch.Tensor

    def __init__(self):
        """Set fake_quant_enabled and PTQ_observer_enabled."""
        super().__init__()
        # fake_quant_enabled and PTQ_observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because NCCL does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('PTQ_observer_enabled', torch.tensor([1], dtype=torch.uint8))

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    @torch.jit.export
    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled = torch.Tensor([1]) if enabled else torch.Tensor([0])

    @torch.jit.export
    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    @torch.jit.export
    def enable_PTQ_observer(self, enabled: bool = True) -> None:
        self.PTQ_observer_enabled = torch.Tensor([1]) if enabled else torch.Tensor([0])

    @torch.jit.export
    def disable_PTQ_observer(self):
        self.enable_PTQ_observer(False)

    @classmethod
    def with_args(cls, **kwargs):
        fake_quant_constructor = _with_args(cls, **kwargs)
        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = "torch.ao.quantization.fake_quantize"
        return fake_quant_constructor

class FakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    The qparams of this module can only be updated manually, or via observation.

    The output of this module is given by::

        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    * :attr:`is_dynamic` indicates whether the fake quantie is a placeholder for dynamic quantization
      operators (choose_qparams -> q -> dq) or static quantization operators (q -> dq)

    * :attr:`scale` defines the scale factor used for quantization.

    * :attr:`zero_point` specifies the quantized value to which 0 in floating point maps to

    * :attr:`fake_quant_enabled` controls the application of fake quantization on tensors, note that
      statistics can still be updated.

    * :attr:`PTQ_observer_enabled` controls statistics collection on tensors

    * :attr:`dtype` specifies the quantized dtype that is being emulated with fake-quantization,
        allowable values are torch.qint8 and torch.quint8.

    Args:

        observer (module): Module for observing statistics on input tensors and calculating scale
          and zero-point.
        observer_kwargs (optional): Arguments for the observer module

    Attributes:
        activation_post_process (Module): User provided module that collects statistics on the input tensor and
          provides a method to calculate scale and zero-point.

    """

    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=None, quant_max=None, is_dynamic=False, **observer_kwargs):
        super().__init__()
        # Populate quant_min/quant_max to observer_kwargs if valid
        if quant_min is not None and quant_max is not None:
            assert quant_min <= quant_max, \
                'quant_min must be less than or equal to quant_max'
            dtype = observer_kwargs.get("dtype", torch.quint8)
            if hasattr(observer, "p"):
                # In case observer is _PartialWrapper, dtype can be stored in
                # observer.p.keywords["dtype"]
                dtype = getattr(getattr(observer, "p", {}), "keywords", {}).get(
                    "dtype", dtype
                )
            assert torch.iinfo(dtype).min <= quant_min, 'quant_min out of bound'
            assert quant_max <= torch.iinfo(dtype).max, 'quant_max out of bound'
            observer_kwargs.update({"quant_min": quant_min, "quant_max": quant_max})
        observer_kwargs["is_dynamic"] = is_dynamic
        self.activation_post_process = observer(**observer_kwargs)
        # TODO: keeping self.quant_min/max for BC; remove after a couple releases
        # Users should use self.activation_post_process.quant_min
        self.quant_min = self.activation_post_process.quant_min
        self.quant_max = self.activation_post_process.quant_max
        self.is_dynamic = self.activation_post_process.is_dynamic
        if _is_float_qparams(self.activation_post_process.qscheme):
            zero_point_dtype = torch.float
        else:
            zero_point_dtype = torch.int
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        self.register_buffer('zero_point', torch.tensor([0], dtype=zero_point_dtype))
        self.dtype = self.activation_post_process.dtype
        self.qscheme = self.activation_post_process.qscheme
        self.ch_axis = self.activation_post_process.ch_axis \
            if hasattr(self.activation_post_process, 'ch_axis') else -1
        assert _is_per_channel(self.qscheme) or \
            _is_per_tensor(self.qscheme), \
            'Only per channel and per tensor quantization are supported in fake quantize' + \
            ' got qscheme: ' + str(self.qscheme)
        self.is_per_channel = _is_per_channel(self.qscheme)
        self.set_forward_call()


    @torch.jit.export
    def calculate_qparams(self):
        return self.activation_post_process.calculate_qparams()

    def __setattr__(self, name, value):
        """
        We overwrite the `__setattr__` method, to intervene and update the forward call if any of the
        buffer values have been updated, before continuing as usual.
        """

        # Call the original __setattr__ method
        super(self.__class__, self).__setattr__(name, value)

        # If any of the buffers has been affected, we assume it is a change to the quantization
        # forward call state. We call the `set_forward_call` method.
        if name in self._buffers:
            # NOTE: make sure this doesn't get called every time qparams get changed, that would suck.
            self.set_forward_call()

    #########################
    # FORWARD METHOD SETTER #
    #########################
    def set_forward_call(self):
        """
        Updates the forward call, based on the values in the int-state toggles/buffers.
        """
        # NOTE: maybe have fake-quant toggle play a role here too, and do away with
        # `_get_quant_or_dequant_forward`.
        if self.PTQ_observer_enabled:
            # PTQ is enabled, and one either returns a quantized or dequantized forward call
            # depending on `self.fake_quant_enabled`.
            self.forward = self._get_PTQ_forward()
        else:
            # No PTQ, only the quantized or dequantized forward call.
            self.forward = self._get_quant_or_dequant_forward()

    ##########################
    # FORWARD METHOD GETTERS #
    ##########################
    def _get_float_forward(self):
        """
        Returns the floating point (dequantized) forward call.
        """
        return self.float_forward

    def _get_fake_quant_forward(self):
        """
        Returns the fake-quantize forward call, depending om the qscheme.
        """
        if _is_per_channel(self.qscheme):
            return self.fake_quant_per_channel_forward
        elif _is_per_tensor(self.qscheme):
            return self.fake_quant_per_tensor_forward
        else:
            raise NotImplementedError("FakeQuantize only supports per-channel or per-tensor quantization")

    def _get_quant_or_dequant_forward(self):
        """
        Returns either the floating-point or fake-quantized forward call,
        depending on whether fake-quantization is enabled or not.
        """
        if self.fake_quant_enabled:
            return self._get_fake_quant_forward()
        else:
            return self._get_float_forward()

    def _get_PTQ_forward(self):
        """
        Returns the PTQ forward call, and it returns either quantized or dequantized tensors.
        """
        if self.fake_quant_enabled:
            base_forward = self._get_fake_quant_forward()
            def PTQ_then_fake_quant_return(X):
                X = self.PTQ_forward(X)
                X = base_forward(X)
                return X
            return PTQ_then_fake_quant_return
        else:
            base_forward = self._get_float_forward()
            def PTQ_then_dequantized_return(X):
                X = self.PTQ_forward(X)
                X = base_forward(X)
                return X
            return PTQ_then_dequantized_return

    ###################
    # FORWARD METHODS #
    ###################
    def float_forward(self, X):
        """
        This is the forward call that returns the input, i.e. returns a
        dequantized tensor and does not update the qparams by omitting any
        quantization operations.
        """
        return X

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

    # Dummy forward call to allow intiailization
    def forward(self, X):
        pass


    @torch.jit.export
    def extra_repr(self):
        return 'forward_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, ' \
               'quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.forward.__name__, self.fake_quant_enabled, self.PTQ_observer_enabled,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)
        destination[prefix + 'scale'] = self.scale
        destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class FixedQParamsFakeQuantize(FakeQuantize):
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
        return 'forward_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, scale={}, zero_point={}, ' \
               'dtype={}, quant_min={}, quant_max={}, qscheme={}'.format(
                   self.forward.__name__, self.fake_quant_enabled, self.PTQ_observer_enabled,
                   self.scale, self.zero_point, self.dtype,
                   self.activation_post_process.quant_min, self.activation_post_process.quant_max, self.qscheme)


class FusedMovingAvgObsFakeQuantize(FakeQuantize):
    r"""Define a fused module to observe the tensor.

    Fused module that is used to observe the input tensor (compute min/max), compute
    scale/zero_point and fake_quantize the tensor.
    This module uses calculation similar MovingAverageMinMaxObserver for the inputs,
    to compute the min/max values in order to compute the scale/zero_point.
    The qscheme input in the observer is used to differentiate between symmetric/affine
    quantization scheme.

    The output of this module is given by
    x_out = (clamp(round(x/scale + zero_point), quant_min, quant_max)-zero_point)*scale

    Similar to :class:`~torch.ao.quantization.FakeQuantize`, and accepts the same attributes as the
    base class.

    """

    def __init__(
        self,
        observer: Any = MovingAverageMinMaxObserver,
        quant_min: int = 0,
        quant_max: int = 255,
        **observer_kwargs: Any
    ) -> None:
        super().__init__(observer, quant_min, quant_max, **observer_kwargs)
        assert isinstance(self.activation_post_process, (MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver)), \
            "Fused observer+fake_quant module only works with MovingAverageMinMaxObserver"
        self.register_buffer("fake_quant_enabled", torch.tensor([1], dtype=torch.long))
        self.register_buffer("PTQ_observer_enabled", torch.tensor([1], dtype=torch.long))
        self.is_symmetric_quant = _is_symmetric_quant(self.activation_post_process.qscheme)

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.activation_post_process.calculate_qparams()

    @torch.jit.export
    def extra_repr(self) -> str:
        return (
            "forward_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, scale={}, zero_point={}, "
            "dtype={}, quant_min={}, quant_max={}, qscheme={}, reduce_range={}".format(
                self.forward.__name__,
                self.fake_quant_enabled,
                self.PTQ_observer_enabled,
                self.scale,
                self.zero_point,
                self.dtype,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
                self.qscheme,
                self.activation_post_process.reduce_range,
            )
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.fused_moving_avg_obs_fake_quant(
            X,
            self.PTQ_observer_enabled,
            self.fake_quant_enabled,
            self.activation_post_process.min_val,
            self.activation_post_process.max_val,
            self.scale,
            self.zero_point,
            self.activation_post_process.averaging_constant,
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
            self.ch_axis,
            self.is_per_channel,
            self.is_symmetric_quant,
        )

default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)
"""
Default fake_quant for activations.
"""

default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
"""
Default fake_quant for weights.
Observer is memoryless since averaging_constant is 1.
"""

default_dynamic_fake_quant = FakeQuantize.with_args(
    observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, is_dynamic=True,
    dtype=torch.quint8, averaging_constant=1)
"""
Default dynamic fake_quant for activations.
"""

default_fixed_qparams_range_neg1to1_fake_quant = (
    FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_neg1to1_observer)
)
default_fixed_qparams_range_0to1_fake_quant = (
    FixedQParamsFakeQuantize.with_args(observer=default_fixed_qparams_range_0to1_observer)
)
# TODO: the following 2 variables are kept for backwards compatibility; remove after a few releases
default_symmetric_fixed_qparams_fake_quant = default_fixed_qparams_range_neg1to1_fake_quant
default_affine_fixed_qparams_fake_quant = default_fixed_qparams_range_0to1_fake_quant

default_per_channel_weight_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
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
default_embedding_fake_quant = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
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

default_embedding_fake_quant_4bit = FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                           qscheme=torch.per_channel_affine_float_qparams,
                                                           ch_axis=0,
                                                           dtype=torch.quint4x2,
                                                           averaging_constant=1)

default_histogram_fake_quant = FakeQuantize.with_args(observer=HistogramObserver,
                                                      quant_min=0,
                                                      quant_max=255,
                                                      dtype=torch.quint8,
                                                      qscheme=torch.per_tensor_affine,
                                                      reduce_range=True)
"""
Fake_quant for activations using a histogram..
"""


default_fused_act_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                       quant_min=0,
                                                                       quant_max=255,
                                                                       dtype=torch.quint8,)

"""
Fused version of `default_fake_quant`, with improved performance.
"""


default_fused_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                      quant_min=-128,
                                                                      quant_max=127,
                                                                      dtype=torch.qint8,
                                                                      qscheme=torch.per_tensor_symmetric)
"""
Fused version of `default_weight_fake_quant`, with improved performance.
"""

default_fused_per_channel_wt_fake_quant = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                                                                  quant_min=-128,
                                                                                  quant_max=127,
                                                                                  dtype=torch.qint8,
                                                                                  qscheme=torch.per_channel_symmetric)
"""
Fused version of `default_per_channel_weight_fake_quant`, with improved performance.
"""

fused_wt_fake_quant_range_neg_127_to_127 = FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                   quant_min=-127,
                                                                                   quant_max=127,
                                                                                   dtype=torch.qint8,
                                                                                   qscheme=torch.per_tensor_symmetric,
                                                                                   eps=2 ** -12)
"""
Fused version of `default_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""

fused_per_channel_wt_fake_quant_range_neg_127_to_127 = \
    FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver,
                                            quant_min=-127,
                                            quant_max=127,
                                            dtype=torch.qint8,
                                            qscheme=torch.per_channel_symmetric,
                                            eps=2 ** -12)

"""
Fused version of `default_per_channel_weight_fake_quant`, with the 8-bit values restricted to [-127, +127], excluding -128.
"""


def _is_fake_quant_script_module(mod):
    """Return true if given mod is an instance of FakeQuantize script module."""
    if isinstance(mod, torch.jit.RecursiveScriptModule):
        # qualified name looks like '__torch__.torch.ao.quantization.fake_quantize.___torch_mangle_2.FakeQuantize'
        suffix = mod._c.qualified_name.split('.', 1)[1]
        name = re.sub(r'\.___torch_mangle_\d+', '', suffix)
        return name == 'torch.ao.quantization.fake_quantize.FakeQuantize' or \
            name == 'torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize'
    return False

#######################################
# QUANTIZATION STATE TOGGLING METHODS #
#######################################
def disable_fake_quant(mod):
    """Disable fake quantization for the module.

    Disable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_fake_quant()

def enable_fake_quant(mod):
    """Enable fake quantization for the module.

    Enable fake quantization for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_fake_quant)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_fake_quant()

def disable_PTQ_observer(mod):
    """Disable observation for this module.

    Disable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.disable_PTQ_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.disable_PTQ_observer()

def enable_PTQ_observer(mod):
    """Enable observation for this module.

    Enable observation for this module, if applicable. Example usage::

      # model is any PyTorch model
      model.apply(torch.ao.quantization.enable_PTQ_observer)

    """
    if isinstance(mod, FakeQuantizeBase) or _is_fake_quant_script_module(mod):
        mod.enable_PTQ_observer()