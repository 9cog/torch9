# torch9 Quick Start Guide

Welcome to torch9! This guide will help you get started with the unified PyTorch domain libraries monorepo.

## Installation

### Basic Installation

Install torch9 with basic dependencies:

```bash
pip install torch9
```

### Domain-Specific Installation

Install with specific domain libraries:

```bash
# Audio processing
pip install torch9[audio]

# Computer vision
pip install torch9[vision]

# Natural Language Processing
pip install torch9[text]

# Reinforcement Learning
pip install torch9[rl]

# Multiple domains
pip install torch9[audio,vision,text]

# All domains
pip install torch9[all]
```

### Development Installation

For contributing or development:

```bash
git clone https://github.com/9cog/torch9.git
cd torch9
pip install -e ".[dev,all]"
```

## Quick Examples

### Audio Processing

```python
from torch9 import audio

# Load audio file
waveform, sample_rate = audio.load_audio("speech.wav")
print(f"Audio shape: {waveform.shape}, Sample rate: {sample_rate}")

# Apply transformation
transform = audio.AudioTransform(sample_rate=16000)
processed = transform(waveform)

# Save processed audio
audio.save_audio("output.wav", processed, sample_rate)
```

### Computer Vision

```python
from torch9 import vision

# Load image
image = vision.load_image("photo.jpg", size=(224, 224))
print(f"Image shape: {image.shape}")

# Use pre-trained model
model = vision.ResNet(num_layers=50, pretrained=True)
features = model(image)

# Apply transformations
transform = vision.ImageTransform(size=(224, 224))
transformed = transform(image)
```

### Text Processing

```python
from torch9 import text

# Tokenize text
tokenizer = text.Tokenizer(vocab_size=10000, max_length=512)
tokens = tokenizer.encode("Hello, world!")
print(f"Tokens: {tokens}")

# Decode tokens
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Create dataset
dataset = text.TextDataset(["sentence 1", "sentence 2", "sentence 3"])
print(f"Dataset size: {len(dataset)}")
```

### Reinforcement Learning

```python
from torch9 import rl

# Create environment
env = rl.Environment(env_name="CartPole-v1")
state = env.reset()

# Define policy
policy = rl.Policy(state_dim=4, action_dim=2)

# Create and train agent
agent = rl.Agent(policy, env)
agent.train(num_episodes=1000)

# Run episode
state = env.reset()
total_reward = 0
for step in range(100):
    action = policy(state)
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break
    state = next_state
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
user_ids = [1, 2, 3, 4, 5]
item_ids = [10, 20, 30, 40, 50]
scores = model.predict(user_ids, item_ids)
print(f"Prediction scores: {scores}")

# Get embeddings
user_embeddings = model.user_embeddings([1, 2, 3])
print(f"User embeddings shape: {user_embeddings.shape}")
```

### LLM Fine-tuning

```python
from torch9 import tune

# Configure fine-tuning
config = tune.LoRAConfig(
    rank=8,
    alpha=16,
    dropout=0.1
)

# Create fine-tuner
tuner = tune.FineTuner("llama-7b", config=config)

# Fine-tune model
metrics = tuner.train(
    dataset=my_dataset,
    num_epochs=3,
    batch_size=8,
    learning_rate=2e-5
)

print(f"Training metrics: {metrics}")

# Evaluate
results = tuner.evaluate(test_dataset)
print(f"Evaluation results: {results}")
```

### Data Loading

```python
from torch9 import data

# Create data pipeline
pipeline = data.DataPipeline(my_data)
pipeline = (pipeline
    .map(lambda x: x * 2)
    .filter(lambda x: x > 0)
    .batch(32))

# Iterate through pipeline
for batch in pipeline:
    process_batch(batch)

# Use data loader
loader = data.DataLoader(
    dataset=my_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in loader:
    train_on_batch(batch)
```

### Media Codec

```python
from torch9 import codec

# Decode video
frames, audio, metadata = codec.decode_video("video.mp4")
print(f"Frames shape: {frames.shape}")

# Decode audio
waveform = codec.decode_audio("audio.mp3")
print(f"Audio shape: {waveform.shape}")

# Use decoders directly
video_decoder = codec.VideoDecoder("video.mp4", device="cuda")
frames = video_decoder.decode(start_frame=0, num_frames=100)

audio_decoder = codec.AudioDecoder("audio.mp3", device="cpu")
waveform = audio_decoder.decode(start_time=0.0, duration=10.0)
```

## Multimodal Example

Combining multiple domains:

```python
import torch
from torch9 import audio, vision, text, data

# Process audio
audio_waveform, sr = audio.load_audio("speech.wav")
audio_features = audio.AudioTransform()(audio_waveform)

# Process image
image = vision.load_image("scene.jpg")
visual_features = vision.ResNet(num_layers=50)(image)

# Process text
text_input = "Description of the scene"
text_tokens = text.tokenize(text_input)

# Combine features
combined_features = torch.cat([
    audio_features.flatten(),
    visual_features.flatten(),
    torch.tensor(text_tokens, dtype=torch.float32)
])

# Create data pipeline
dataset = [(audio_features, visual_features, text_tokens)]
pipeline = data.DataPipeline(dataset).batch(8)
```

## Running Examples

The repository includes working examples:

```bash
# Multimodal integration example
python examples/multimodal_example.py

# Reinforcement learning example
python examples/rl_example.py

# Recommender system example
python examples/recommender_example.py
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=torch9 --cov-report=html

# Specific module tests
pytest tests/test_audio.py -v
pytest tests/test_vision.py -v
```

## Common Issues

### Import Error

If you get import errors, make sure you've installed the required domain:

```bash
pip install torch9[audio]  # For audio module
pip install torch9[all]    # For all modules
```

### Missing Dependencies

Install development dependencies:

```bash
pip install torch9[dev]
```

## Next Steps

- Read the [Architecture Documentation](ARCHITECTURE.md)
- Explore the [examples/](../examples/) directory
- Check out the [API Reference](API.md)
- Join the community discussions

## Getting Help

- GitHub Issues: https://github.com/9cog/torch9/issues
- Documentation: https://github.com/9cog/torch9#readme

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

BSD 3-Clause License - see [LICENSE](../LICENSE) for details.
