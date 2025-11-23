# Torch Organization Repositories

This directory contains all repositories from the original Torch organization (https://github.com/torch), integrated into the torch9 monorepo as a coherent whole.

## Overview

These repositories represent the original Torch7 framework and its ecosystem - a scientific computing framework with wide support for machine learning algorithms that puts GPUs first. Torch is based on Lua and LuaJIT.

## Integration Details

- **Integration Date**: November 23, 2025
- **Source Organization**: https://github.com/torch
- **Total Repositories**: 46
- **Integration Method**: Direct clone without submodules (all .git directories removed)

## Repository List

### Core Framework
- **torch7** - The main Torch7 framework
- **distro** - Torch installation in a self-contained folder
- **TH** - Standalone C TH library
- **luajit-rocks** - LuaJIT and luarocks in one location

### Neural Networks
- **nn** - Neural network package for Torch7
- **nngraph** - Graph computation for nn
- **rnn** - Torch recurrent neural networks
- **cunn** - CUDA implementation of neural network modules
- **cutorch** - A CUDA backend for Torch7

### Utilities
- **xlua** - A set of useful functions to extend Lua (string, table, ...)
- **sys** - A system utility package for Torch
- **paths** - Path manipulation utilities
- **class** - Oriented Object Programming for Lua
- **env** - Sets up default torch environment
- **cwrap** - C wrapper for Torch
- **argcheck** - A powerful argument checker and function overloading system
- **trepl** - A pure Lua-based, lightweight REPL for Torch

### Optimization & Training
- **optim** - A numeric optimization package for Torch
- **threads** - Threads for Lua and LuaJIT with transparent data exchange

### Data Structures
- **tds** - Torch C data structures
- **vector** - Vector utilities
- **hash** - Hashing functions for Torch7

### Graphics & Visualization
- **image** - An Image toolbox for Torch
- **gnuplot** - Gnuplot interface for Torch
- **qtlua** - Lua interface to QT library
- **qttorch** - QT bindings for Torch
- **cairo-ffi** - LuaJIT FFI interface to Cairo Graphics
- **sdl2-ffi** - A LuaJIT interface to SDL2

### Documentation & Web
- **torch.github.io** - Torch's web page
- **dok** - Documentation system for Torch
- **rocks** - Rocks for Torch
- **tutorials** - A series of machine learning tutorials for Torch7
- **demos** - Demos and tutorials around Torch7

### FFI & Bindings
- **ffi** - FFI bindings for Torch7 (LuaJIT-speed access to Tensors and Storages)
- **sundown-ffi** - A LuaJIT interface to the Sundown library (Markdown implementation)

### NLP & Text
- **senna** - NLP SENNA interface to LuaJIT

### Networking
- **socketfile** - Adds file-over-sockets support for Torch
- **graph** - Graph package for Torch

### Testing
- **testme** - Unit Testing for Torch
- **torchunit** - Unit testing framework

### Installation & Administration
- **ezinstall** - One-line install scripts for Torch
- **luarocks-mirror** - Luarocks mirror (because luarocks.org is not completely reliable!)
- **nimbix-admin** - Utility scripts for start/stop/ssh to nimbix instances

### Utilities (Additional)
- **rational** - Rational numbers for Lua
- **xt** - Torch TH/THC c++11 wrapper

### Deprecated
- **DEPRECEATED-torch7-distro** - Old Torch7 distribution (deprecated)

## Usage

These repositories are integrated as source code into the monorepo. They can be:
- Referenced for historical purposes
- Used as a foundation for new work
- Studied to understand the original Torch7 architecture
- Migrated or adapted for modern PyTorch applications

## License

Each repository maintains its original license. Please refer to the LICENSE file in each subdirectory for specific licensing information.

## Links

- Original Torch Organization: https://github.com/torch
- Torch7 Main Repository: https://github.com/torch/torch7
- PyTorch (successor): https://pytorch.org/

## Notes

This is a snapshot of the Torch organization repositories integrated into the torch9 monorepo. The repositories are no longer git submodules and have had their .git directories removed to create a coherent, unified codebase.
