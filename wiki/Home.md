# torch9 Wiki - Complete Documentation

Welcome to the torch9 documentation wiki! This repository contains structured documentation for the entire torch9 project, including both modern PyTorch domain libraries and the legacy Torch7 framework.

## Table of Contents

### 00. Home
- [README](00-Home/README.md) - Main project README
- [Completion Summary](00-Home/COMPLETION_SUMMARY.md) - Project completion status
- [Torch Integration Summary](00-Home/TORCH_INTEGRATION_SUMMARY.md) - Details on Torch7 integration
- [Copilot Memories](00-Home/copilot-memories.md) - Development notes

### 01. Getting Started
- [Quick Start Guide](01-Getting-Started/QUICKSTART.md) - Get up and running quickly with torch9

### 02. Architecture
- [Architecture Overview](02-Architecture/ARCHITECTURE.md) - System architecture and design decisions

### 03. Torch7 Legacy Framework

#### Core Framework
- [Torch Overview](03-Torch7-Legacy/Torch-Overview.md) - Overview of Torch7 framework
- [Core Documentation](03-Torch7-Legacy/core/) - Torch7 core library documentation
  - [Index](03-Torch7-Legacy/core/index.md)
  - [Tensors](03-Torch7-Legacy/core/tensor.md)
  - [Storage](03-Torch7-Legacy/core/storage.md)
  - [File I/O](03-Torch7-Legacy/core/file.md)
  - [Serialization](03-Torch7-Legacy/core/serialization.md)
  - [Math Operations](03-Torch7-Legacy/core/maths.md)
  - [Random Numbers](03-Torch7-Legacy/core/random.md)
  - [Command Line](03-Torch7-Legacy/core/cmdline.md)
  - [Timer](03-Torch7-Legacy/core/timer.md)
  - [Tester](03-Torch7-Legacy/core/tester.md)
  - [Utilities](03-Torch7-Legacy/core/utility.md)
  - [ROADMAP](03-Torch7-Legacy/core/ROADMAP.md)

#### Neural Networks
- [Neural Networks (nn)](03-Torch7-Legacy/neural-networks/) - Core neural network library
  - [Index](03-Torch7-Legacy/neural-networks/index.md)
  - [Module](03-Torch7-Legacy/neural-networks/module.md) - Base module class
  - [Containers](03-Torch7-Legacy/neural-networks/containers.md) - Sequential, Parallel, etc.
  - [Simple Layers](03-Torch7-Legacy/neural-networks/simple.md) - Basic layer types
  - [Convolution](03-Torch7-Legacy/neural-networks/convolution.md) - Convolutional layers
  - [Table Layers](03-Torch7-Legacy/neural-networks/table.md) - Table-based layers
  - [Transfer Functions](03-Torch7-Legacy/neural-networks/transfer.md) - Activation functions
  - [Training](03-Torch7-Legacy/neural-networks/training.md) - Training procedures
  - [Criterion](03-Torch7-Legacy/neural-networks/criterion.md) - Loss functions
  - [Testing](03-Torch7-Legacy/neural-networks/testing.md) - Testing utilities
  - [Overview](03-Torch7-Legacy/neural-networks/overview.md) - Complete overview

- [nngraph](03-Torch7-Legacy/neural-networks/nngraph/) - Graph-based neural networks
  - [README](03-Torch7-Legacy/neural-networks/nngraph/README.md)

- [RNN](03-Torch7-Legacy/neural-networks/rnn/) - Recurrent Neural Networks
  - [README](03-Torch7-Legacy/neural-networks/rnn/README.md)
  - [Recurrent Modules](03-Torch7-Legacy/neural-networks/rnn/recurrent.md)
  - [Sequencer](03-Torch7-Legacy/neural-networks/rnn/sequencer.md)
  - [Criterion](03-Torch7-Legacy/neural-networks/rnn/criterion.md)
  - [REINFORCE](03-Torch7-Legacy/neural-networks/rnn/reinforce.md)
  - [Miscellaneous](03-Torch7-Legacy/neural-networks/rnn/miscellaneous.md)

- [CUNN](03-Torch7-Legacy/neural-networks/cunn/) - CUDA Neural Networks
  - [README](03-Torch7-Legacy/neural-networks/cunn/README.md)
  - [CUNN Modules](03-Torch7-Legacy/neural-networks/cunn/cunnmodules.md)

#### Optimization
- [Optimization (optim)](03-Torch7-Legacy/optimization/) - Optimization algorithms
  - [Introduction](03-Torch7-Legacy/optimization/intro.md)
  - [Algorithms](03-Torch7-Legacy/optimization/algos.md) - SGD, Adam, etc.
  - [Logger](03-Torch7-Legacy/optimization/logger.md)

#### Image Processing
- [Image Processing](03-Torch7-Legacy/image-processing/) - Image manipulation and processing
  - [Index](03-Torch7-Legacy/image-processing/index.md)
  - [Loading/Saving](03-Torch7-Legacy/image-processing/saveload.md)
  - [Tensor Construction](03-Torch7-Legacy/image-processing/tensorconstruct.md)
  - [Simple Transforms](03-Torch7-Legacy/image-processing/simpletransform.md)
  - [Parametric Transforms](03-Torch7-Legacy/image-processing/paramtransform.md)
  - [Color Space](03-Torch7-Legacy/image-processing/colorspace.md)
  - [Drawing](03-Torch7-Legacy/image-processing/drawing.md)
  - [GUI](03-Torch7-Legacy/image-processing/gui.md)

#### GUI and Visualization
- [gnuplot](03-Torch7-Legacy/gui-visualization/gnuplot/) - Plotting with gnuplot
  - [Index](03-Torch7-Legacy/gui-visualization/gnuplot/index.md)
  - [Plot Line](03-Torch7-Legacy/gui-visualization/gnuplot/plotline.md)
  - [Plot Matrix](03-Torch7-Legacy/gui-visualization/gnuplot/plotmatrix.md)
  - [Plot Histogram](03-Torch7-Legacy/gui-visualization/gnuplot/plothistogram.md)
  - [Plot Surface](03-Torch7-Legacy/gui-visualization/gnuplot/plotsurface.md)
  - [Plot 3D Points](03-Torch7-Legacy/gui-visualization/gnuplot/plot3dpoints.md)
  - [File Operations](03-Torch7-Legacy/gui-visualization/gnuplot/file.md)
  - [Common Options](03-Torch7-Legacy/gui-visualization/gnuplot/common.md)
  - [Custom Plots](03-Torch7-Legacy/gui-visualization/gnuplot/custom.md)

- [qtlua](03-Torch7-Legacy/gui-visualization/qtlua/) - Qt-based GUI for Lua
  - [Index](03-Torch7-Legacy/gui-visualization/qtlua/index.md)
  - [Qt Core](03-Torch7-Legacy/gui-visualization/qtlua/qtcore.md)
  - [Qt GUI](03-Torch7-Legacy/gui-visualization/qtlua/qtgui.md)
  - [Qt Widgets](03-Torch7-Legacy/gui-visualization/qtlua/qtwidget.md)
  - [Qt SVG](03-Torch7-Legacy/gui-visualization/qtlua/qtsvg.md)
  - [Qt IDE](03-Torch7-Legacy/gui-visualization/qtlua/qtide.md)
  - [Qt UI Loader](03-Torch7-Legacy/gui-visualization/qtlua/qtuiloader.md)

- [qttorch](03-Torch7-Legacy/gui-visualization/qttorch/) - Qt integration for Torch

#### Utilities
- [paths](03-Torch7-Legacy/utilities/paths/) - Path manipulation
  - [Index](03-Torch7-Legacy/utilities/paths/index.md)
  - [Directory Paths](03-Torch7-Legacy/utilities/paths/dirpaths.md)
  - [Directory Functions](03-Torch7-Legacy/utilities/paths/dirfunctions.md)
  - [File Names](03-Torch7-Legacy/utilities/paths/filenames.md)
  - [Miscellaneous](03-Torch7-Legacy/utilities/paths/misc.md)

- [cwrap](03-Torch7-Legacy/utilities/cwrap/) - C wrapping utility
  - [Index](03-Torch7-Legacy/utilities/cwrap/index.md)
  - [High Level Interface](03-Torch7-Legacy/utilities/cwrap/highlevelinterface.md)
  - [Argument Types](03-Torch7-Legacy/utilities/cwrap/argumenttypes.md)
  - [User Types](03-Torch7-Legacy/utilities/cwrap/usertypes.md)
  - [Example](03-Torch7-Legacy/utilities/cwrap/example.md)

- [argcheck](03-Torch7-Legacy/utilities/argcheck/) - Argument checking
- [trepl](03-Torch7-Legacy/utilities/trepl/) - Interactive REPL
- [sys](03-Torch7-Legacy/utilities/sys/) - System utilities
- [xlua](03-Torch7-Legacy/utilities/xlua/) - Extended Lua utilities
- [ffi](03-Torch7-Legacy/utilities/ffi/) - Foreign Function Interface
- [tds](03-Torch7-Legacy/utilities/tds/) - Torch Data Structures
- [hash](03-Torch7-Legacy/utilities/hash/) - Hashing utilities
- [threads](03-Torch7-Legacy/utilities/threads/) - Multi-threading support
- [graph](03-Torch7-Legacy/utilities/graph/) - Graph utilities
- [class](03-Torch7-Legacy/utilities/class/) - Class system
- [rational](03-Torch7-Legacy/utilities/rational/) - Rational numbers
- [vector](03-Torch7-Legacy/utilities/vector/) - Vector operations
- [xt](03-Torch7-Legacy/utilities/xt/) - Extended tensors

#### Tutorials and Examples
- [Torch Tutorials](03-Torch7-Legacy/tutorials/torch-tutorials/) - Official tutorials
- [Demos](03-Torch7-Legacy/tutorials/demos/) - Demo applications
  - [Main README](03-Torch7-Legacy/tutorials/demos/README.md)
  - [Attention Demo](03-Torch7-Legacy/tutorials/demos/attention/)
  - [Person Detector](03-Torch7-Legacy/tutorials/demos/person-detector/)
  - [Tracker](03-Torch7-Legacy/tutorials/demos/tracker/)

### 04. Torch Subprojects

Additional components and tools:

- [cutorch](04-Torch-Subprojects/cutorch/) - CUDA support for Torch
- [TH](04-Torch-Subprojects/TH/) - Tensor library (C backend)
- [distro](04-Torch-Subprojects/distro/) - Torch distribution
- [ezinstall](04-Torch-Subprojects/ezinstall/) - Easy installation scripts
- [luajit-rocks](04-Torch-Subprojects/luajit-rocks/) - LuaJIT and LuaRocks
- [rocks](04-Torch-Subprojects/rocks/) - Package management
- [senna](04-Torch-Subprojects/senna/) - NLP tools
- [cairo-ffi](04-Torch-Subprojects/cairo-ffi/) - Cairo graphics FFI
- [sdl2-ffi](04-Torch-Subprojects/sdl2-ffi/) - SDL2 FFI
- [sundown-ffi](04-Torch-Subprojects/sundown-ffi/) - Markdown parser FFI
- [socketfile](04-Torch-Subprojects/socketfile/) - Socket file utilities
- [torchunit](04-Torch-Subprojects/torchunit/) - Unit testing framework
- [testme](04-Torch-Subprojects/testme/) - Testing utilities
- [nimbix-admin](04-Torch-Subprojects/nimbix-admin/) - Nimbix cloud administration
- [dok](04-Torch-Subprojects/dok/) - Documentation generation
- [torch.github.io](04-Torch-Subprojects/torch-github-io/) - Official website documentation
- [DEPRECATED-torch7-distro](04-Torch-Subprojects/DEPRECATED-torch7-distro/) - Deprecated distribution

### 05. Contributing

Guidelines for contributing to torch9 and its components:

- [torch7 Contributing](05-Contributing/torch7-CONTRIBUTING.md)
- [nn Contributing](05-Contributing/nn-CONTRIBUTING.md)
- [cunn Contributing](05-Contributing/cunn-CONTRIBUTING.md)
- [cutorch Contributing](05-Contributing/cutorch-CONTRIBUTING.md)

### 06. Blog Posts

Historical blog posts and articles from the Torch community:

- [Blog Posts](06-Blog-Posts/_posts/) - Collection of blog posts about Torch7

## Navigation Tips

- Use the directory structure above to navigate to specific topics
- All original README files from each component are preserved
- Documentation maintains its original structure within each category
- Cross-references between documents are maintained where possible

## Additional Resources

- [Main Project Repository](https://github.com/9cog/torch9)
- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [Legacy Torch7 Website](http://torch.ch/) (archived)

---

*This wiki provides comprehensive documentation for both modern PyTorch domain libraries and the complete legacy Torch7 framework.*
