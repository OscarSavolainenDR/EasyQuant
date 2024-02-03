# BetterQuant

This is an extension to PyTorch's native neural network quantization library. 

## Why BetterQuant?
I started building this because I struggled with quantization code, and became an expert at dealing with its intricacies and silent failing modes, but nobody should have to go through that. Quantization should Just Work TM, and it should be quick and easy! This is ridiculously easy to get started with, just pip install, import, and it'll immediately all work with your PyTorch code.

## Features
This library has cool soome features, such as:
- Automated Graph FX Mode quantization, out of the box, simply.
- Enable fusing of any layer, and any activation. You want a quantizable `ConvPReLU`? You got a quantizable `ConvPReLU`!
- Improve the speed of the Quantization Aware Training (QAT) and Post-Training Quantization (PTQ) forward calls.
- Improve to-console logging of quantization objects.
- A huge suite of PTQ observers, e.g. fixed weight observers, clamp-to-some-percentile observers, KL-divergence minimizing observers, etc.
- Never-seen-before quantization-specific visualization and analysis tools, such as activation histograms overlain with the quantization grid, and Jacobian Sensisity Analysis.
- Super-fast, memory-safe custom backend kernels written in Rust, instead of Torch's native C++. One can also select to use Torch's native C++ backend kernels, where they are available.
- All of the standard quantization biolerplate is simplified assignign of qconfigs, fusing, PTQ, QAT, you name it.

It all Just Works TM, and it's all quick and easy to use!

## Getting started

## How to Contribute


