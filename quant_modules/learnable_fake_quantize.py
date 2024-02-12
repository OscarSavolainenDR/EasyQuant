import torch
from torch.ao.quantization._learnable_fake_quantize import _LearnableFakeQuantize as TorchLearnableFakeQuantize
from torch.nn.parameter import Parameter
from typing import List

__all__: List[str] = []

class BQLearnableFakeQuantize(TorchLearnableFakeQuantize):
    r"""This is a wrapper around Torch's `_LearnableFakeQuantize` class. It modularizes the forward calls,
    speeding up their execution and allowing for the scalable introduction of different `observer`
    and `transform` algorithms.

    `observer_enabled` is renamed to the more descriptive `PTQ_observer_enabled`, also
    enabling the introduction of observers that are not PTQ. `static_enabled` is fully eliminated,
    as it is redundant to `observer_enabled`.

    The qparams of this quantization module can only be updated manually, via observation, or via
    learning via gradient descent.

    The output of this module is given by::
        x_out = (
          clamp(round(x/scale + zero_point), quant_min, quant_max) - zero_point
        ) * scale

    """
    def __init__(self, observer, quant_min=0, quant_max=255, scale=1., zero_point=0., channel_len=-1,
                 use_grad_scaling=False, **observer_kwargs):
        super().__init__(observer=observer, quant_min=quant_min, quant_max=quant_max, scale=scale,
                         zero_point=zero_point, channel_len=channel_len, use_grad_scaling=use_grad_scaling,
                         **observer_kwargs)
        delattr(self, 'observer_enabled')
        self.register_buffer('PTQ_observer_enabled', torch.tensor([0], dtype=torch.uint8))
        self.observer = self._get_PTQ_forward()
        self.transform = self._get_fake_quant_forward()
        # The symmetric/affine and grad-scaling/no-grad-scaling forward calls
        # do not typically change over the lifetime of the quantization module.
        self.symmetric_or_affine = self._get_symmetric_or_affine_forward()
        self.grad_scaling = self._get_grad_scaling()

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enable parameter learning over static PTQ observer estimates.

        Enables learning of quantization parameters and
        disables PTQ observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True)
        self.enable_fake_quant()
        self.disable_PTQ_observer()
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        r"""Enable static estimates of quantization parameters.

        Enables static PTQ observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False)
        self.enable_fake_quant()
        self.enable_PTQ_observer()

    @torch.jit.export
    def enable_static_observation(self):
        r"""Enable accumulation of data without updating quantization parameters.

        Enables static PTQ observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False)
        self.disable_fake_quant()
        self.enable_PTQ_observer()

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        r"""
        Toggles qparam gradients on/off for this quantization module instance.
        """
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled

    @torch.jit.export
    def observe_quant_params(self):
        print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
        print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')

    @torch.jit.export
    def calculate_qparams(self):
        """
        Calculates the 'true' qarams, i.e. clamps the scale to a minimum allowed value
        and clamps and rounds the zero-point to be an integer between `self.quant_min`
        and `self.quant_max`. The latter is because the zero-point is treated as a continuous
        value in the state-dict but in inference should be an appropriate integer.
        """
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    def forward(self, X):
        r"""
        The Learnable Fake Quantize forward call. This first calls the observer (or doesn't,
        depending on if PTQ is activated or not). It then does some affine or symmetric
        specific ops, calculates the grad factor, and then fake-quantizes the input tensor X.
        """
        self.observer(X)
        self.symmetric_or_affine()
        grad_factor = self.grad_scaling(X)
        X = self.transform(X, grad_factor)
        return X

    ##########################
    # FORWARD METHOD SETTERS #
    ##########################
    # We overwrite the state toggling methods. Not that the `fake_quant_enabled`
    # and `PTQ_observer_enabled` togglews are now vestigial and not key to the
    # nature of the forward call of `self`.
    def enable_fake_quant(self) -> None:
        """
        We overwrite the method so that it sets the `self.transform` forward call
        as the fake-quant forward call.
        """
        self.fake_quant_enabled[0] = 1
        self.transform = self._get_fake_quant_forward()

    def disable_fake_quant(self) -> None:
        """
        We overwrite the method so that it sets the `self.transform` forward call
        as the dequantized forward call.
        """
        self.fake_quant_enabled[0] = 0
        self.transform = self._get_dummy_forward()

    def enable_PTQ_observer(self) -> None:
        """
        We overwrite the method so that it sets the `self.observer` forward call
        as the PTQ forward call.
        """
        self.PTQ_observer_enabled[0] = 1
        self.observer = self._get_PTQ_forward()

    def disable_PTQ_observer(self) -> None:
        """
        We overwrite the method so that it sets the `self.observer` forward call
        as the dequantized forward call.
        """
        self.PTQ_observer_enabled[0] = 0
        self.observer = self._get_observer_dummy_forward()

    ############################
    # OBSERVER FORWARD GETTERS #
    ############################
    def _get_observer_dummy_forward(self):
        """
        Returns the dummy (no observation) observer call
        """
        return self.dummy_observer_forward

    def _get_PTQ_forward(self):
        """
        Returns the PTQ forward call.
        """
        return self.PTQ_forward

    ###############################
    # GRADSCALING FORWARD GETTERS #
    ###############################
    def _get_grad_scaling(self):
        """
        We return either the grad-scaling calculation, or a function that returns a default
        grad-scaling factor of 1, depending on if we do grad-sclaing or not in this module.
        """
        if self.grad_scaling:
            return self.grad_scaling_forward
        else:
            return self.no_grad_scaling_forward

    #############################
    # TRANSFORM FORWARD GETTERS #
    #############################
    def _get_dummy_forward(self):
        """
        This returns the dummy forward, which in turn just returns the input.
        """
        return self.dummy_forward

    def _get_fake_quant_forward(self):
        """
        We return the fake-quant forward call, either per-tensor or per-channel.
        """
        if self.qscheme in (torch.per_channel_symmetric, torch.per_channel_affine):
            return self.fake_quant_per_channel_forward
        elif self.fake_quant_per_tensor_forward:
            return self.fake_quant_per_tensor_forward
        else:
            raise NotImplementedError("_LearnableFakeQuantize only supports per-tensor and per-channel quantization")

    def _get_symmetric_or_affine_forward(self):
        """
        Depending on if we are doing symmetric or affine quantization, we return
        the `center_zero_symmetric` operation or the `dummy` operation.
        """
        if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
            return self.symmetric_specific_ops
        elif self.qscheme in (torch.per_channel_affine, torch.per_tensor_affine):
            return self.affine_specific_ops

    ###########################
    # OBSERVER FORWARD METHODS #
    ############################
    def dummy_observer_forward(self, X):
        """
        The dummy observer call. It merely clamps the qparam scale.
        TODO: move this clamp elsewhere, i.e. into the fake_quant forward.
        """
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

    def PTQ_forward(self, X):
        """
        This forward call performs PTQ observation on the tensor, updating
        the qparams of `self`, i.e. the quantization module.
        """
        # Calls PTQ observer
        self.activation_post_process(X.detach())
        # Returns the qparams calculated by the PTQ observer
        _scale, _zero_point = self.activation_post_process.calculate_qparams()
        # Overwrites the scale and zero-point of `self`
        _scale = _scale.to(self.scale.device)
        _zero_point = _zero_point.to(self.zero_point.device)
        self.scale.data.copy_(_scale)
        self.zero_point.data.copy_(_zero_point)

    #############################
    # TRANSFORM FORWARD METHODS #
    ##############################
    def dummy_forward(self, X):
        """
        This is a dummy forward that just returns the input. In the context of
        the transform, it is equivelent to not quantizing the input tensor in any
        way and keep it dequantized.
        """
        return X

    def grad_scaling_forward(self, X):
        """
        Returns a `grad_factor` for the QAT grad scaling.
        """
        # NOTE: we could maybe pre-calculate some of this, or maybe all of it, depending
        # on if the input to this module has a consistent shape.
        grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
        return grad_factor

    def no_grad_scaling_forward(self, X):
        """
        Returns a `grad_factor` of 1.
        """
        return 1.0

    def symmetric_specific_ops(self):
        """
        For the symmetric fake-quant calls, we center the zero-points.
        TODO: We should delete the zero-point as a param and set as a buffer.
        """
        self.zero_point.data.zero_()

    def affine_specific_ops(self):
        """
        There are no affine-specific ops.
        """
        pass

    def fake_quant_per_channel_forward(self, X, grad_factor):
        """
        The per-channel fake quant forward call.
        """
        return torch._fake_quantize_learnable_per_channel_affine(
                X, self.scale, self.zero_point, self.ch_axis,
                self.quant_min, self.quant_max, grad_factor)

    def fake_quant_per_tensor_forward(self, X, grad_factor):
        """
        The per-channel fake quant forward call.
        """
        return torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)
    ############
    # PRINTING #
    ############
    @torch.jit.export
    def extra_repr(self):
        return 'observer_call={}, transform_call={}, fake_quant_enabled={}, PTQ_observer_enabled={}, ' \
               'grad_scaling={}, quant_min={}, quant_max={}, dtype={}, qscheme={}, ch_axis={}, ' \
               'scale={}, zero_point={}'.format(
                   self.observer.__name__, self.transform.__name__, self.fake_quant_enabled, self.PTQ_observer_enabled,
                   self.use_grad_scaling, self.activation_post_process.quant_min, self.activation_post_process.quant_max,
                   self.dtype, self.qscheme, self.ch_axis, self.scale, self.zero_point)



