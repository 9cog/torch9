# torch9 - Unified PyTorch Domain Libraries Monorepo

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A cohesive monorepo integrating all major PyTorch domain libraries into a unified, easy-to-use package. Also includes the complete original Torch7 framework and ecosystem (46 repositories) for historical reference and legacy support.

## 🌟 Overview

`torch9` brings together the entire PyTorch ecosystem of domain-specific libraries under one roof, providing a seamless experience for deep learning practitioners working across multiple domains.

### Integrated Libraries

#### Modern PyTorch Domain Libraries

- **🔊 torch9.audio** - Audio signal processing and deep learning (torchaudio)
- **👁️ torch9.vision** - Computer vision models and transformations (torchvision)
- **📝 torch9.text** - Natural Language Processing utilities (torchtext)
- **🎮 torch9.rl** - Reinforcement Learning tools (torchrl)
- **🎯 torch9.rec** - Recommender systems at scale (torchrec)
- **🔧 torch9.tune** - Fine-tuning workflows for LLMs (torchtune)
- **📊 torch9.data** - Flexible data loading pipelines (torchdata)
- **🎬 torch9.codec** - Fast media encoding/decoding (torchcodec)

#### Original Torch7 Framework

The monorepo also includes all 46 repositories from the original Torch organization, providing:
- **📚 Historical Reference** - Complete Torch7 framework source code
- **🔧 Legacy Support** - Original Lua-based machine learning framework
- **🎓 Learning Resource** - Understanding the evolution from Torch7 to PyTorch

For details, see [torch/README.md](torch/README.md).

## 🚀 Installation

### Basic Installation

```bash
pip install torch9
```

### Install with Specific Domain Libraries

```bash
# Install with audio support
pip install torch9[audio]

# Install with vision support
pip install torch9[vision]

# Install with text/NLP support
pip install torch9[text]

# Install with RL support
pip install torch9[rl]

# Install all domain libraries
pip install torch9[all]

# Install with development tools
pip install torch9[dev]
```

## 📖 Quick Start

### Audio Processing

```python
from torch9 import audio

# Load and process audio
waveform, sample_rate = audio.load_audio("speech.wav")

# Apply transformations
transform = audio.AudioTransform(sample_rate=16000)
processed = transform(waveform)

# Save processed audio
audio.save_audio("output.wav", processed, sample_rate)
```

### Computer Vision

```python
from torch9 import vision

# Load and transform images
image = vision.load_image("cat.jpg", size=(224, 224))

# Use pre-trained models
model = vision.ResNet(num_layers=50, pretrained=True)
features = model(image)

# Apply transformations
transform = vision.ImageTransform(size=(224, 224))
transformed = transform(image)
```

### Natural Language Processing

```python
from torch9 import text

# Tokenize text
tokenizer = text.Tokenizer(vocab_size=10000)
tokens = text.tokenize("Hello, world!", tokenizer)

# Work with text datasets
dataset = text.TextDataset(["sentence 1", "sentence 2"])
print(f"Dataset size: {len(dataset)}")
```

### Reinforcement Learning

```python
from torch9 import rl

# Create RL environment
env = rl.Environment(env_name="CartPole-v1")

# Define policy
policy = rl.Policy(state_dim=4, action_dim=2)

# Train agent
agent = rl.Agent(policy, env)
agent.train(num_episodes=1000)
```

### Recommender Systems

```python
from torch9 import rec

# Create recommender model
model = rec.RecommenderModel(
    num_users=10000,
    num_items=5000,
    embedding_dim=128
)

# Make predictions
scores = model.predict(user_ids=[1, 2], item_ids=[10, 20])
```

### LLM Fine-tuning

```python
from torch9 import tune

# Configure fine-tuning
config = tune.LoRAConfig(rank=8, alpha=16)
tuner = tune.FineTuner("llama-7b", config=config)

# Fine-tune model
metrics = tuner.train(
    dataset=my_dataset,
    num_epochs=3,
    batch_size=8
)

# Evaluate
results = tuner.evaluate(test_dataset)
```

### Data Loading

```python
from torch9 import data

# Create flexible data pipeline
pipeline = data.DataPipeline(my_data)
pipeline = (pipeline
    .map(lambda x: x * 2)
    .filter(lambda x: x > 0)
    .batch(32))

# Use enhanced data loader
loader = data.DataLoader(
    dataset=my_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

### Media Codec

```python
from torch9 import codec

# Decode video
frames, audio, metadata = codec.decode_video("video.mp4")

# Decode audio
waveform = codec.decode_audio("audio.mp3")

# Use decoders directly
video_decoder = codec.VideoDecoder("video.mp4", device="cuda")
frames = video_decoder.decode(start_frame=0, num_frames=100)
```

## 🏗️ Monorepo Structure

```
torch9/
├── packages/
│   └── torch9/
│       ├── __init__.py
│       ├── audio/          # Audio processing
│       ├── vision/         # Computer vision
│       ├── text/           # NLP utilities
│       ├── rl/             # Reinforcement learning
│       ├── rec/            # Recommender systems
│       ├── tune/           # LLM fine-tuning
│       ├── data/           # Data loading
│       └── codec/          # Media encoding/decoding
├── torch/                  # Original Torch7 framework (46 repos)
│   ├── torch7/             # Main Torch7 framework
│   ├── nn/                 # Neural networks
│   ├── optim/              # Optimization algorithms
│   ├── image/              # Image processing
│   ├── tutorials/          # Torch7 tutorials
│   └── ...                 # 41+ more repositories
├── tests/                  # Unit tests
├── examples/               # Usage examples
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── setup.py                # Setup script
```

## 🧪 Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/9cog/torch9.git
cd torch9

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Format code
black packages/
isort packages/

# Type checking
mypy packages/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=torch9 --cov-report=html

# Run specific test file
pytest tests/test_audio.py
```

## 🤝 Contributing

Contributions are welcome! This monorepo aims to provide a cohesive experience across all PyTorch domain libraries. Please:

1. Follow the existing code style (Black formatter, isort)
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This monorepo integrates and builds upon the excellent work of the PyTorch team and community across multiple domain libraries:

- PyTorch Core Team
- torchaudio contributors
- torchvision contributors
- torchtext contributors
- torchrl contributors
- torchrec contributors
- torchtune contributors
- torchdata contributors
- torchcodec contributors

## 📚 Documentation

For detailed documentation on each submodule:

- [Audio Documentation](docs/audio.md)
- [Vision Documentation](docs/vision.md)
- [Text Documentation](docs/text.md)
- [RL Documentation](docs/rl.md)
- [Rec Documentation](docs/rec.md)
- [Tune Documentation](docs/tune.md)
- [Data Documentation](docs/data.md)
- [Codec Documentation](docs/codec.md)

## 🔗 Links

- [GitHub Repository](https://github.com/9cog/torch9)
- [PyTorch Official](https://pytorch.org/)
- [PyTorch Domains](https://pytorch.org/domains/)

---

Made with ❤️ by the torch9 community