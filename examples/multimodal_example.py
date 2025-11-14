"""Example: Using multiple torch9 modules together."""

import torch

from torch9 import audio, data, text, vision


def multimodal_example():
    """Demonstrate integration of audio, vision, and text modules."""

    print("=" * 60)
    print("torch9 Multimodal Integration Example")
    print("=" * 60)

    # Audio processing
    print("\n1. Audio Processing")
    print("-" * 40)
    waveform, sr = audio.load_audio("sample.wav")
    print(f"Loaded audio: shape={waveform.shape}, sample_rate={sr}")

    audio_transform = audio.AudioTransform(sample_rate=sr)
    processed_audio = audio_transform(waveform)
    print(f"Processed audio: shape={processed_audio.shape}")

    # Vision processing
    print("\n2. Vision Processing")
    print("-" * 40)
    image = vision.load_image("sample.jpg", size=(224, 224))
    print(f"Loaded image: shape={image.shape}")

    model = vision.ResNet(num_layers=50, pretrained=False)
    features = model(image)
    print(f"Extracted features: shape={features.shape}")

    # Text processing
    print("\n3. Text Processing")
    print("-" * 40)
    sample_text = "This is a multimodal deep learning example using torch9"
    tokenizer = text.Tokenizer(vocab_size=10000)
    tokens = tokenizer.encode(sample_text)
    print(f"Original text: {sample_text}")
    print(f"Tokens: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")

    # Data pipeline
    print("\n4. Data Pipeline")
    print("-" * 40)
    sample_data = list(range(100))
    pipeline = data.DataPipeline(sample_data)
    pipeline = pipeline.map(lambda x: x * 2).filter(lambda x: x < 100).batch(10)

    print("Created data pipeline with map, filter, and batch operations")
    batch_count = 0
    for batch in pipeline:
        batch_count += 1
    print(f"Processed {batch_count} batches")

    print("\n" + "=" * 60)
    print("All modules working together successfully!")
    print("=" * 60)


if __name__ == "__main__":
    multimodal_example()
