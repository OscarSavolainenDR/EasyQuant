# oscars-torch-int8-extension

This is an extension to PyTorch's native int8 quantization library. 

It does various things, such as:
- Improve the speed of the Quantization Aware Training (QAT) and Post-Training Quantization (PTQ) forward calls.
- Improve to-console logging of quantization objects.
- Implement a larger suite of PTQ observers, e.g. fixed weight observers, clamp-to-some-percentile observers, KL-divergence minimizing observers, etc.
- Include various visualization and analysis tools, such as activation histograms overlain with the quantization grid, and Jacobian Sensisity Analysis overlain with the quantization grid.
- Writes the backend kernels in Rust, instead of Torch's native C++. One can also select to use Torch's native C++ backend kernels, where they are available.

Qconfig assignment, PTQ and QAT are provided as modularized functions for ease of use.

We also provide a custom model-loading function, necessary to work with the quantization forward calls.
