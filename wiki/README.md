# torch9 Wiki Documentation

This directory contains all documentation from the torch9 repository, organized in a structured format suitable for GitHub wiki or standalone documentation hosting.

## Directory Structure

```
wiki/
├── Home.md                          # Main index and navigation
├── 00-Home/                         # Project overview
│   ├── README.md                    # Main project README
│   ├── COMPLETION_SUMMARY.md        # Completion status
│   ├── TORCH_INTEGRATION_SUMMARY.md # Torch7 integration details
│   └── copilot-memories.md          # Development notes
├── 01-Getting-Started/              # Quick start guides
│   └── QUICKSTART.md                # Getting started guide
├── 02-Architecture/                 # Architecture documentation
│   └── ARCHITECTURE.md              # System architecture
├── 03-Torch7-Legacy/                # Legacy Torch7 framework
│   ├── Torch-Overview.md            # Torch7 overview
│   ├── core/                        # Core library docs
│   ├── neural-networks/             # Neural network docs
│   │   ├── *.md                     # Main nn docs
│   │   ├── nngraph/                 # Graph-based networks
│   │   ├── rnn/                     # Recurrent networks
│   │   └── cunn/                    # CUDA networks
│   ├── optimization/                # Optimization algorithms
│   ├── image-processing/            # Image manipulation
│   ├── gui-visualization/           # GUI and plotting
│   │   ├── gnuplot/                 # Gnuplot interface
│   │   ├── qtlua/                   # Qt interface
│   │   └── qttorch/                 # Qt-Torch integration
│   ├── utilities/                   # Utility libraries
│   │   ├── paths/                   # Path manipulation
│   │   ├── cwrap/                   # C wrapping
│   │   ├── argcheck/                # Argument checking
│   │   ├── trepl/                   # REPL
│   │   ├── sys/                     # System utilities
│   │   ├── xlua/                    # Extended Lua
│   │   ├── ffi/                     # Foreign Function Interface
│   │   ├── tds/                     # Data structures
│   │   ├── hash/                    # Hashing
│   │   ├── threads/                 # Threading
│   │   ├── graph/                   # Graphs
│   │   ├── class/                   # Class system
│   │   ├── rational/                # Rational numbers
│   │   ├── vector/                  # Vectors
│   │   └── xt/                      # Extended tensors
│   └── tutorials/                   # Tutorials and demos
│       ├── torch-tutorials/         # Official tutorials
│       └── demos/                   # Demo applications
├── 04-Torch-Subprojects/            # Additional components
│   ├── cutorch/                     # CUDA support
│   ├── TH/                          # Tensor library
│   ├── distro/                      # Distribution
│   ├── ezinstall/                   # Installation
│   ├── luajit-rocks/                # LuaJIT/LuaRocks
│   ├── rocks/                       # Package management
│   ├── senna/                       # NLP
│   ├── cairo-ffi/                   # Cairo FFI
│   ├── sdl2-ffi/                    # SDL2 FFI
│   ├── sundown-ffi/                 # Markdown FFI
│   ├── socketfile/                  # Socket utilities
│   ├── torchunit/                   # Testing
│   ├── testme/                      # Testing utilities
│   ├── nimbix-admin/                # Cloud admin
│   ├── dok/                         # Doc generation
│   ├── torch-github-io/             # Website docs
│   └── DEPRECATED-torch7-distro/    # Deprecated
├── 05-Contributing/                 # Contributing guides
│   ├── torch7-CONTRIBUTING.md
│   ├── nn-CONTRIBUTING.md
│   ├── cunn-CONTRIBUTING.md
│   └── cutorch-CONTRIBUTING.md
└── 06-Blog-Posts/                   # Historical blog posts
    └── _posts/                      # Blog post collection
```

## How to Use This Wiki

### For GitHub Wiki

To use this structure with GitHub Wiki:

1. Clone this wiki directory
2. Navigate to your repository's Wiki tab
3. Create pages matching the structure above
4. Copy content from the corresponding files

### For Standalone Documentation

This structure works well with static site generators:

- **MkDocs**: Place files in `docs/` and configure `mkdocs.yml`
- **Docusaurus**: Use as source for documentation pages
- **Jekyll**: Compatible with GitHub Pages
- **GitBook**: Import as a GitBook project
- **Sphinx**: Can be adapted for Sphinx documentation

### Navigation

Start with `Home.md` which provides:
- Complete table of contents
- Navigation links to all sections
- Overview of all available documentation

## Documentation Coverage

This wiki includes documentation for:

✅ **Modern torch9 Project**
- Main README and project overview
- Architecture documentation
- Quick start guides
- Integration summaries

✅ **Torch7 Legacy Framework (46+ repositories)**
- Core tensor library
- Neural networks (nn, nngraph, rnn, cunn)
- Optimization algorithms
- Image processing
- GUI and visualization tools
- Extensive utility libraries
- Tutorials and examples

✅ **Contributing Guidelines**
- Multiple component contribution guides
- Development workflows

✅ **Historical Content**
- Blog posts
- Community articles
- Website documentation

## File Formats

All documentation is in Markdown format (`.md`) with occasional:
- PNG images (diagrams, screenshots)
- Plain text files (`.txt`) for specific configurations

## Updates and Maintenance

This wiki was generated from the source repository and includes all documentation as of the copy date. To update:

1. Pull latest changes from main repository
2. Re-run the documentation copy process
3. Review and merge changes

## Contributing

To improve this documentation:

1. Edit source files in the main repository
2. Submit pull requests with documentation improvements
3. Documentation changes will be included in next wiki update

## License

All documentation follows the licensing of the torch9 project and its constituent components. See LICENSE files in respective component directories.

---

**Note**: This wiki represents a complete snapshot of all documentation from the torch9 monorepo, including both modern PyTorch domain libraries and the complete legacy Torch7 framework.
