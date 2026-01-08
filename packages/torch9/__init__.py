"""
torch9 - Unified monorepo for all torch/* domain libraries

This package provides a cohesive integration of PyTorch domain libraries including:
- audio: Audio signal processing and deep learning
- vision: Computer vision models and transformations
- text: Natural Language Processing utilities
- rl: Reinforcement Learning tools
- rec: Recommender systems
- tune: Fine-tuning workflows for large models
- data: Flexible data loading pipelines
- codec: Fast media encoding/decoding
- logic: Tensor validation, type checking, and logical operations
- algo: Master algorithm framework for composable ML pipelines
"""

__version__ = "0.1.0"

__all__ = [
    # Domain libraries
    "audio",
    "vision",
    "text",
    "rl",
    "rec",
    "tune",
    "data",
    "codec",
    # Core frameworks
    "logic",
    "algo",
]


def __getattr__(name: str):
    """Lazy import submodules to avoid unnecessary dependencies."""
    import importlib

    if name in __all__:
        # Dynamically import the submodule
        module = importlib.import_module(f".{name}", __name__)
        # Cache it in globals to avoid re-importing
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
