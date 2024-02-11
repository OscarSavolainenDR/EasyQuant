# BetterQuant

BetterQuant is an extension to PyTorch's native neural network quantization library. 

## Why BetterQuant?
Neural network quantization is difficult. The frameworks aren't easy to navigate, things will fail silently, and good luck creating anything custom. I became an expert at dealing with its intricacies and pain points, but nobody should have to go through that. Quantization should all Just Work :tm:, and it should be quick and easy! And so was born BetterQuant! It is ridiculously easy to get started with, just pip install, import, and it'll immediately all interface seamlessly with your PyTorch code.

## Features
This library has cool soome features, such as:
- Automated Graph FX Mode quantization, out of the box, simply.
- Enable quantization-aware fusing of any layer, and any activation. You want a `ConvPReLU`? You got a `ConvPReLU`!
- Significantly improve the speed of the Quantization Aware Training (QAT) and Post-Training Quantization (PTQ) forward calls. Save GPU time, save money!
- A huge suite of PTQ observers, e.g. fixed weight observers, clamp-to-some-percentile observers, KL-divergence minimizing observers, etc.
- Never-seen-before quantization-specific visualization and analysis tools: see how your tensor interacts with its quantization grid, and see what parts of your quantization range matter the most with our Jacobian Sensitivity Analysis plots.
- Super-fast, memory-safe custom backend kernels written in Rust, instead of Torch's native C++. One can also select to use Torch's native C++ backend kernels, where they are available.
- All of the standard quantization boilerplate is simplified and modularised, e.g. assigning of qconfigs, fusing, PTQ, QAT, you name it.
- An extensive array of unit tests to catch your deadly silent errors, and make them loud :fireworks:.

It all works out the box, and it's all quick and easy to use!

## Getting started

## How to Contribute


