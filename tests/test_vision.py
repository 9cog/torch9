"""Tests for torch9.vision module."""

import pytest
import torch
import warnings


class TestImageTransform:
    """Tests for ImageTransform class."""

    def test_init_default(self):
        """Test ImageTransform initialization with defaults."""
        from torch9.vision import ImageTransform

        transform = ImageTransform()
        assert transform.size == (224, 224)

    def test_init_custom_size(self):
        """Test ImageTransform initialization with custom size."""
        from torch9.vision import ImageTransform

        transform = ImageTransform(size=(128, 128))
        assert transform.size == (128, 128)

    def test_call_returns_input(self):
        """Test ImageTransform __call__ returns input unchanged."""
        from torch9.vision import ImageTransform

        transform = ImageTransform()
        image = torch.randn(3, 224, 224)
        result = transform(image)
        assert torch.equal(result, image)

    def test_repr(self):
        """Test ImageTransform string representation."""
        from torch9.vision import ImageTransform

        transform = ImageTransform(size=(224, 224))
        assert "ImageTransform" in repr(transform)


class TestCompose:
    """Tests for Compose class."""

    def test_compose_multiple_transforms(self):
        """Test Compose with multiple transforms."""
        from torch9.vision import Compose, ImageTransform

        transform = Compose([ImageTransform(), ImageTransform()])
        image = torch.randn(3, 224, 224)
        result = transform(image)
        assert result.shape == image.shape


class TestResize:
    """Tests for Resize class."""

    def test_init(self):
        """Test Resize initialization."""
        from torch9.vision import Resize

        resize = Resize((128, 128))
        assert resize.size == (128, 128)

    def test_resize_image(self):
        """Test Resize actually resizes."""
        from torch9.vision import Resize

        resize = Resize((64, 64))
        image = torch.randn(3, 128, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = resize(image)
        assert result.shape == (3, 64, 64)


class TestCenterCrop:
    """Tests for CenterCrop class."""

    def test_init(self):
        """Test CenterCrop initialization."""
        from torch9.vision import CenterCrop

        crop = CenterCrop((64, 64))
        assert crop.size == (64, 64)

    def test_center_crop(self):
        """Test CenterCrop crops correctly."""
        from torch9.vision import CenterCrop

        crop = CenterCrop((64, 64))
        image = torch.randn(3, 128, 128)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = crop(image)
        assert result.shape == (3, 64, 64)


class TestNormalize:
    """Tests for Normalize class."""

    def test_init(self):
        """Test Normalize initialization."""
        from torch9.vision import Normalize

        norm = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        assert norm.mean == (0.5, 0.5, 0.5)
        assert norm.std == (0.5, 0.5, 0.5)

    def test_normalize(self):
        """Test Normalize changes values."""
        from torch9.vision import Normalize

        norm = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        image = torch.ones(3, 64, 64) * 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = norm(image)
        # (0.5 - 0.5) / 0.5 = 0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


class TestResNet:
    """Tests for ResNet class."""

    def test_init(self):
        """Test ResNet initialization."""
        from torch9.vision import ResNet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ResNet(num_layers=50, pretrained=False)
        assert model.num_layers == 50
        assert model.pretrained is False

    def test_forward(self):
        """Test ResNet forward pass."""
        from torch9.vision import ResNet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ResNet(num_layers=18, pretrained=False)
            x = torch.randn(1, 3, 224, 224)
            result = model(x)
        assert isinstance(result, torch.Tensor)


class TestVGG:
    """Tests for VGG class."""

    def test_init(self):
        """Test VGG initialization."""
        from torch9.vision import VGG

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VGG(num_layers=16, pretrained=False)
        assert model.num_layers == 16
        assert model.pretrained is False


class TestEfficientNet:
    """Tests for EfficientNet class."""

    def test_init(self):
        """Test EfficientNet initialization."""
        from torch9.vision import EfficientNet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = EfficientNet(version="b0", pretrained=False)
        assert model.version == "b0"
        assert model.pretrained is False


class TestGetModel:
    """Tests for get_model function."""

    def test_get_resnet(self):
        """Test get_model for ResNet."""
        from torch9.vision import get_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = get_model("resnet18", pretrained=False)
        assert model.num_layers == 18

    def test_get_vgg(self):
        """Test get_model for VGG."""
        from torch9.vision import get_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = get_model("vgg16", pretrained=False)
        assert model.num_layers == 16


class TestListModels:
    """Tests for list_models function."""

    def test_returns_list(self):
        """Test list_models returns a list."""
        from torch9.vision import list_models

        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert "resnet50" in models


class TestLoadImage:
    """Tests for load_image function."""

    def test_returns_tensor(self):
        """Test load_image returns tensor."""
        from torch9.vision import load_image

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = load_image("test.jpg")
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3  # RGB channels

    def test_with_size(self):
        """Test load_image with specific size."""
        from torch9.vision import load_image

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = load_image("test.jpg", size=(128, 128))
        assert image.shape == (3, 128, 128)


class TestSaveImage:
    """Tests for save_image function."""

    def test_no_exception(self):
        """Test save_image doesn't raise exception."""
        from torch9.vision import save_image

        image = torch.rand(3, 64, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should not raise exception even in fallback mode
            save_image("output.png", image)


class TestUtilities:
    """Tests for utility functions."""

    def test_is_torchvision_available(self):
        """Test is_torchvision_available returns boolean."""
        from torch9.vision import is_torchvision_available

        result = is_torchvision_available()
        assert isinstance(result, bool)
