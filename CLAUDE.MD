# CLAUDE.MD - torch9 Monorepo

## Project Overview

**torch9** is a unified monorepo integrating all major PyTorch domain libraries into a single cohesive package, plus the complete original Torch7 framework (46 repositories) for historical reference.

### Repository URL
https://github.com/9cog/torch9

## Quick Reference

### Key Directories
```
packages/torch9/          # Main Python package (8 domain modules)
torch/                    # Original Torch7 framework (46 repos, ~274MB)
tests/                    # Test suite (44 tests, 80% coverage)
examples/                 # Usage examples (3 files)
docs/                     # Documentation (ARCHITECTURE.md, QUICKSTART.md)
```

### Core Package Structure
```
packages/torch9/
├── __init__.py           # Main entry with lazy loading
├── audio/                # Audio processing (torchaudio)
├── vision/               # Computer vision (torchvision)
├── text/                 # NLP utilities (torchtext)
├── rl/                   # Reinforcement learning (torchrl)
├── rec/                  # Recommender systems (torchrec)
├── tune/                 # LLM fine-tuning (torchtune)
├── data/                 # Data loading (torchdata)
└── codec/                # Media codec (torchcodec)
```

## Development Commands

### Installation
```bash
pip install -e ".[dev,all]"    # Full dev setup
pip install -e ".[audio]"      # With specific domain
```

### Testing
```bash
pytest tests/ -v --cov=torch9  # Run all tests with coverage
pytest tests/test_audio.py     # Run specific test file
```

### Code Quality
```bash
black packages/ examples/ tests/    # Format code
isort packages/ examples/ tests/    # Sort imports
flake8 packages/torch9              # Lint
mypy packages/                      # Type check
```

## Code Style

- **Formatter**: Black (line-length=100)
- **Import sorter**: isort (profile=black)
- **Python**: 3.8+ compatible
- **Type hints**: Used throughout

## Architecture Notes

### Lazy Loading Pattern
Submodules are imported on-demand via `__getattr__` in `packages/torch9/__init__.py`:
```python
from torch9 import audio  # Only loads audio module when accessed
```

### Optional Dependencies
Each domain has separate dependencies in `pyproject.toml`:
- `torch9[audio]`, `torch9[vision]`, `torch9[text]`, `torch9[rl]`
- `torch9[all]` for everything
- `torch9[dev]` for development tools

### Current Status
- **Version**: 0.1.0 (Alpha)
- **Tests**: 44 tests, 100% passing
- **Coverage**: 80%
- **Implementations**: Placeholder/stub implementations for demonstration

## Current Development Phase

### Completed
1. Monorepo structure with 8 integrated domain modules
2. Lazy loading architecture
3. Comprehensive test suite (44 tests)
4. CI/CD pipeline (GitHub Actions)
5. Documentation (README, ARCHITECTURE, QUICKSTART)
6. Original Torch7 integration (46 repos)
7. Examples for multimodal, RL, and recommender workflows

### Next Phase: Production Integration
Priority tasks from COMPLETION_SUMMARY.md:

1. **Replace placeholder implementations** with actual torch/* library calls
   - Start with `audio/` and `vision/` modules (most mature upstream libraries)
   - Wire up actual torchaudio/torchvision functions

2. **Add real pre-trained models** and weights support
   - Model registry in each domain module
   - Weight downloading utilities

3. **Implement proper data loaders** from each library
   - Leverage torchdata patterns
   - Support streaming and iterable datasets

4. **Improve test coverage to 95%+**
   - Add integration tests for actual library calls
   - Add edge case tests

5. **Add benchmarking suite** for performance validation

6. **Create migration guides** from individual packages to torch9

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, dependencies, tool settings |
| `packages/torch9/__init__.py` | Main package entry, lazy loading |
| `tests/conftest.py` | Shared pytest fixtures |
| `.github/workflows/tests.yml` | CI/CD pipeline |
| `COMPLETION_SUMMARY.md` | Implementation status and next steps |
| `TORCH_INTEGRATION_SUMMARY.md` | Torch7 integration details |

## Testing Approach

Each module has its own test file:
- `test_torch9.py` - Core package tests (11 tests)
- `test_audio.py` - Audio module tests (5 tests)
- `test_vision.py` - Vision module tests (6 tests)
- `test_text.py` - Text module tests (6 tests)
- `test_rl.py` - RL module tests (7 tests)
- `test_data.py` - Data module tests (9 tests)

Run tests before committing:
```bash
pytest tests/ -v
```

## CI/CD Pipeline

GitHub Actions runs on push/PR:
- **OS**: Ubuntu, macOS, Windows
- **Python**: 3.8, 3.9, 3.10, 3.11
- **Checks**: lint, format, type-check, tests, coverage

## Common Tasks

### Adding a new module feature
1. Edit relevant `packages/torch9/<module>/__init__.py`
2. Add tests in `tests/test_<module>.py`
3. Run `pytest` and `black`/`isort`
4. Update docs if needed

### Integrating actual torch/* library
1. Add real dependency to `pyproject.toml` optional-dependencies
2. Replace stub implementation with actual library imports
3. Add try/except for graceful degradation when dep not installed
4. Update tests to use actual functionality

### Adding examples
1. Create new file in `examples/`
2. Follow existing patterns (multimodal_example.py)
3. Use only torch9 imports, not underlying libraries directly

## Notes

- Torch7 code in `torch/` is historical reference - don't modify
- All 46 Torch7 repos have their .git directories removed
- Current implementations are stubs demonstrating the API surface
- Priority is replacing stubs with real torch/* integrations
