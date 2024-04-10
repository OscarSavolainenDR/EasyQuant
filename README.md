# EasyQuant

EasyQuant is an extension to PyTorch's native neural network quantization library. 

## Why EasyQuant?
Neural network quantization is difficult. The frameworks aren't easy to navigate, and things will often fail silently. I got decent at doing quantization, but it was painful, and I wanted to put together the useful things I've come across along the way. That is the idea behind Easy Quant!

It is super easy to get started with (incredible pun), just pip install, import, and it'll all work out the box with your PyTorch code.

## Features
This library has some cool features, such as:
- Full comparability with Eager, FX Graph, and Export Mode quantization.
- Significantly improve the speed of the Quantization Aware Training (QAT) and Post-Training Quantization (PTQ) forward calls. Save GPU time, save money!
- A large suite of non-natively-supported-by-PyTorch PTQ observers, e.g. fixed weight observers, clamp-to-some-percentile observers, KL-divergence minimizing observers, etc.
- Custom quantization-specific visualization and analysis tools: see how your tensor interacts with its quantization grid, and see what parts of your quantization range matter the most with Jacobian Sensitivity Analysis plots.
- TODO: An extensive array of unit tests to catch your deadly silent errors, and make them loud :fireworks:.


## Getting started
Just clone and pip install (ideally into a venv or conda env):
```
mkdir EasyQuant
cd EasyQuant
git clone git@github.com:OscarSavolainenDR/EasyQuant.git .
pip install -e .
```

## Suggestions
I'd generally suggest adding any folder that you generate plots into (e.g. Sensitivity Analysis plots), into a `.gitignore` file. It'll keep our git history from getting filled up with PNG files.

## How to Contribute
Open an issue, and we'll go from there! This is a very new library.


