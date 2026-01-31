# CLAUDE.MD - torch9 Monorepo

## Project Overview

**torch9** is a unified Python monorepo integrating all major PyTorch domain libraries into a single package, plus the original Torch7 framework (46 repositories) as a historical reference. Version 0.1.0 (Alpha).

Repository: https://github.com/9cog/torch9

## Directory Layout

```
packages/torch9/          # Main Python package (10 modules, ~3,600 LOC)
  __init__.py             # Entry point with lazy loading via __getattr__
  audio/                  # torchaudio integration (MelSpectrogram, MFCC, Resample)
  vision/                 # torchvision integration (ResNet, VGG, EfficientNet, transforms)
  text/                   # torchtext (Tokenizer, TextDataset)
  rl/                     # torchrl (Environment, Policy, Agent)
  rec/                    # torchrec (EmbeddingTable, RecommenderModel)
  tune/                   # torchtune (LoRAConfig, FineTuner)
  data/                   # torchdata (DataPipeline, DataLoader)
  codec/                  # torchcodec (VideoDecoder, AudioDecoder)
  logic/                  # Tensor Logic Framework (TensorSpec, TensorGuard, TensorOps, DeviceManager, MemoryTracker)
  algo/                   # Master Algorithm Framework (Algorithm, Pipeline, Step, Scheduler, Registry, Branch/Merge/Loop)
torch/                    # Original Torch7 (46 repos, ~274MB) - READ ONLY, historical reference
tests/                    # 149 tests across 9 files (conftest.py + 8 test modules)
examples/                 # 3 working examples (multimodal, RL, recommender)
docs/                     # ARCHITECTURE.md, QUICKSTART.md
.github/workflows/        # CI/CD (tests.yml)
```

## Development Commands

### Setup

```bash
pip install -e ".[dev,all]"    # Full dev install (all domains + dev tools)
pip install -e ".[dev]"        # Dev tools only (no domain extras)
pip install -e ".[audio]"      # Specific domain
```

### Running Tests

```bash
# PYTHONPATH is needed so pytest resolves torch9 from packages/
PYTHONPATH=packages pytest tests/ -v                          # All tests
PYTHONPATH=packages pytest tests/ -v --cov=torch9             # With coverage
PYTHONPATH=packages pytest tests/test_audio.py -v             # Single module
PYTHONPATH=packages pytest tests/test_logic.py::test_name -v  # Single test
```

Note: `pyproject.toml` sets `addopts = "-v --cov=torch9 --cov-report=term-missing"`, so coverage runs by default.

### Code Quality

```bash
black packages/ tests/ examples/         # Format (line-length=100)
isort packages/ tests/ examples/         # Sort imports (profile=black)
flake8 packages/torch9                   # Lint (E9,F63,F7,F82 strict; max-complexity=10)
mypy packages/torch9                     # Type check (may have non-blocking errors)
```

### Checking Before Commit (mirrors CI)

```bash
black --check packages/torch9 && isort --check-only packages/torch9 && flake8 packages/torch9 --select=E9,F63,F7,F82 --show-source && PYTHONPATH=packages pytest tests/ -v
```

## Code Style

- **Black**: line-length=100, target py38-py311
- **isort**: profile=black, line_length=100
- **flake8**: strict on E9/F63/F7/F82, advisory max-complexity=10, max-line-length=127
- **mypy**: python_version=3.8, warn_return_any=true, disallow_untyped_defs=false (permissive)
- **Python**: 3.8+ compatible, type hints used throughout
- All tool configs live in `pyproject.toml` (except flake8 which is configured in CI)

## Architecture

### Lazy Loading

Submodules load on first access via `__getattr__` in `packages/torch9/__init__.py`. Importing `torch9` is cheap; domains load only when referenced:

```python
import torch9          # Fast - no domain code loaded
torch9.audio           # Loads audio module on first access
from torch9 import vision  # Same lazy behavior
```

### Optional Dependencies

Each domain has its own extras in `pyproject.toml`. Core requires only `torch>=2.0.0` and `numpy>=1.21.0`. Modules use try/except to fall back gracefully when optional deps are missing (e.g., `is_torchaudio_available()`).

### Package Discovery

setuptools finds packages via `[tool.setuptools.packages.find]` with `where = ["packages"]` and `include = ["torch9*"]`.

## Test Structure

| File | Tests | Scope |
|------|-------|-------|
| `test_torch9.py` | 11 | Core package, lazy loading, version |
| `test_audio.py` | 15 | AudioTransform, MelSpectrogram, MFCC, utilities |
| `test_vision.py` | 22 | ImageTransform, Compose, ResNet, model registry |
| `test_text.py` | 6 | Tokenizer, TextDataset |
| `test_rl.py` | 7 | Environment, Policy, Agent |
| `test_data.py` | 9 | DataPipeline, DataLoader |
| `test_logic.py` | 36 | TensorSpec, ShapeSpec, TensorOps, DeviceManager, MemoryTracker |
| `test_algo.py` | 43 | Algorithm, Step, Pipeline, Scheduler, Registry, flow control |

Shared fixtures in `tests/conftest.py`: `sample_tensor`, `sample_waveform`, `sample_text`, `sample_dataset`.

## CI/CD Pipeline

GitHub Actions (`.github/workflows/tests.yml`) runs on push/PR to `main`/`develop`:
- **Matrix**: Ubuntu + macOS + Windows, Python 3.8/3.9/3.10/3.11
- **Steps**: flake8 lint -> black check -> isort check -> mypy (continue-on-error) -> pytest with coverage -> Codecov upload

## Common Tasks

### Adding a feature to an existing module
1. Edit `packages/torch9/<module>/__init__.py`
2. Add tests in `tests/test_<module>.py`
3. Run `black`, `isort`, then `pytest`

### Replacing a stub with a real torch/* integration
1. Add the real dependency to `[project.optional-dependencies]` in `pyproject.toml`
2. In the module, import the real library inside try/except for graceful fallback
3. Expose a helper like `is_<lib>_available()` to check at runtime
4. Update tests to validate real functionality

### Adding a new example
1. Create `examples/<name>_example.py`
2. Use only `torch9` imports (not underlying libraries directly)
3. Follow patterns from `examples/multimodal_example.py`

## Key Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, all tool configs |
| `packages/torch9/__init__.py` | Package entry, lazy loading, version |
| `tests/conftest.py` | Shared pytest fixtures |
| `.github/workflows/tests.yml` | CI pipeline definition |
| `setup.py` | Minimal wrapper delegating to pyproject.toml |

## Current Status

- **149 tests**, 100% pass rate, 74% coverage
- **audio/vision**: Real torchaudio/torchvision integrations with fallbacks
- **logic/algo**: Full frameworks implemented (~2,000 LOC combined)
- **text/rl/rec/tune/data/codec**: Stub implementations demonstrating API surface
- **Torch7**: 46 repos integrated as historical reference (do not modify)

### Next Steps
- Replace remaining stubs with real torch/* integrations
- Add benchmarking suite
- Improve coverage to 95%+
- Create migration guides from individual packages
