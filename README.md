# oscars-torch-int8-extension
An extension to PyTorch's quantization library, which improves QAT training speed, and gives greater control over the quantization behavior (observers, quantization forward calls, and gradients). The backend is written in Rust, with the optional to use Torch's original native C++ kernels for the subset of Torch-supported forward/backward calls..
