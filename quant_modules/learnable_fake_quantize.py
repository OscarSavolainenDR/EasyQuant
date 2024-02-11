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
                         zero_point=zero_point, channel_len=channel_len, use_grad_sclaing=use_grad_scaling)

    @torch.jit.export
    def enable_param_learning(self):
        r"""Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=True) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=False)
        return self

    @torch.jit.export
    def enable_static_estimate(self):
        """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=True) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def enable_static_observation(self):
        """Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
        self.toggle_qparam_learning(enabled=False) \
            .toggle_fake_quant(enabled=False) \
            .toggle_observer_update(enabled=True)

    @torch.jit.export
    def toggle_observer_update(self, enabled=True):
        self.static_enabled[0] = int(enabled)  # type: ignore[operator]
        return self

    @torch.jit.export
    def enable_observer(self, enabled=True):
        self.toggle_observer_update(enabled)

    @torch.jit.export
    def toggle_qparam_learning(self, enabled=True):
        self.learning_enabled[0] = int(enabled)  # type: ignore[operator]
        self.scale.requires_grad = enabled
        self.zero_point.requires_grad = enabled
        return self

    @torch.jit.export
    def toggle_fake_quant(self, enabled=True):
        self.fake_quant_enabled[0] = int(enabled)
        return self

    @torch.jit.export
    def observe_quant_params(self):
        print(f'_LearnableFakeQuantize Scale: {self.scale.detach()}')
        print(f'_LearnableFakeQuantize Zero Point: {self.zero_point.detach()}')

    @torch.jit.export
    def calculate_qparams(self):
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.detach()
        zero_point = self.zero_point.detach().round().clamp(self.quant_min, self.quant_max).long()
        return scale, zero_point

    def forward(self, X):
        if self.static_enabled[0] == 1:  # type: ignore[index]
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.activation_post_process.calculate_qparams()
            _scale = _scale.to(self.scale.device)
            _zero_point = _zero_point.to(self.zero_point.device)
            self.scale.data.copy_(_scale)
            self.zero_point.data.copy_(_zero_point)
        else:
            self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                self.zero_point.data.zero_()

            if self.use_grad_scaling:
                grad_factor = 1.0 / (X.numel() * self.quant_max) ** 0.5
            else:
                grad_factor = 1.0
            if self.qscheme in (
                    torch.per_channel_symmetric, torch.per_channel_affine):
                X = torch._fake_quantize_learnable_per_channel_affine(
                    X, self.scale, self.zero_point, self.ch_axis,
                    self.quant_min, self.quant_max, grad_factor)
            else:
                X = torch._fake_quantize_learnable_per_tensor_affine(
                    X, self.scale, self.zero_point,
                    self.quant_min, self.quant_max, grad_factor)

        return X