"""
torch9.vision - Computer vision models and transformations

This module provides:
- Popular CNN architectures (ResNet, VGG, EfficientNet, etc.)
- Image transformations and augmentations
- Pre-trained models for classification, detection, segmentation
- Vision datasets and utilities

When torchvision is installed, this module provides full functionality
with access to pre-trained models and optimized transforms.
Without torchvision, basic fallback implementations are available.
"""

import warnings
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

# Try to import torchvision for full functionality
_TORCHVISION_AVAILABLE = False
try:
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.transforms.functional as TF
    from torchvision import models as tv_models
    from torchvision.io import read_image, write_png, write_jpeg

    _TORCHVISION_AVAILABLE = True
except ImportError:
    torchvision = None
    transforms = None
    TF = None
    tv_models = None

__all__ = [
    "load_image",
    "save_image",
    "ImageTransform",
    "Compose",
    "Resize",
    "CenterCrop",
    "Normalize",
    "ToTensor",
    "RandomHorizontalFlip",
    "RandomCrop",
    "ResNet",
    "VGG",
    "EfficientNet",
    "get_model",
    "list_models",
    "is_torchvision_available",
]


def is_torchvision_available() -> bool:
    """Check if torchvision is available."""
    return _TORCHVISION_AVAILABLE


class ImageTransform:
    """
    Base class for image transformations.

    This is a composable transform that can be extended or used as a
    pass-through identity transform.

    Args:
        size: Target size for image (height, width)
    """

    def __init__(self, size: Tuple[int, int] = (224, 224)):
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply transformation to image."""
        return image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class Compose(ImageTransform):
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transforms to compose
    """

    def __init__(self, transform_list: List[Callable]):
        super().__init__()
        self.transforms = transform_list

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class Resize(ImageTransform):
    """
    Resize image to given size.

    Args:
        size: Target size (height, width) or single int for square
        interpolation: Interpolation mode
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        interpolation: str = "bilinear",
    ):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(size)
        self.interpolation = interpolation

        if _TORCHVISION_AVAILABLE:
            interp_mode = getattr(
                torchvision.transforms.InterpolationMode,
                interpolation.upper(),
                torchvision.transforms.InterpolationMode.BILINEAR,
            )
            self._transform = transforms.Resize(size, interpolation=interp_mode)
        else:
            self._transform = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            # Fallback using torch interpolate
            if image.dim() == 3:
                image = image.unsqueeze(0)
                squeezed = True
            else:
                squeezed = False

            result = torch.nn.functional.interpolate(
                image, size=self.size, mode=self.interpolation, align_corners=False
            )

            if squeezed:
                result = result.squeeze(0)
            return result


class CenterCrop(ImageTransform):
    """
    Center crop image to given size.

    Args:
        size: Target size (height, width) or single int for square
    """

    def __init__(self, size: Union[int, Tuple[int, int]]):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(size)

        if _TORCHVISION_AVAILABLE:
            self._transform = transforms.CenterCrop(size)
        else:
            self._transform = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            # Fallback center crop
            h, w = image.shape[-2:]
            th, tw = self.size
            i = (h - th) // 2
            j = (w - tw) // 2
            return image[..., i : i + th, j : j + tw]


class Normalize(ImageTransform):
    """
    Normalize image with mean and std.

    Args:
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel
    """

    def __init__(
        self,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.mean = mean
        self.std = std

        if _TORCHVISION_AVAILABLE:
            self._transform = transforms.Normalize(mean=mean, std=std)
        else:
            self._transform = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            # Fallback normalization
            mean = torch.tensor(self.mean).view(-1, 1, 1)
            std = torch.tensor(self.std).view(-1, 1, 1)
            return (image - mean) / std


class ToTensor(ImageTransform):
    """
    Convert PIL Image or numpy array to tensor.

    Also scales values from [0, 255] to [0.0, 1.0].
    """

    def __init__(self):
        super().__init__()
        if _TORCHVISION_AVAILABLE:
            self._transform = transforms.ToTensor()
        else:
            self._transform = None

    def __call__(self, image: Any) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            # Fallback: assume already tensor or convert numpy
            if isinstance(image, torch.Tensor):
                return image.float() / 255.0 if image.dtype == torch.uint8 else image
            else:
                # Try to convert from numpy
                import numpy as np

                if isinstance(image, np.ndarray):
                    tensor = torch.from_numpy(image)
                    if tensor.dim() == 2:
                        tensor = tensor.unsqueeze(0)
                    elif tensor.dim() == 3 and tensor.shape[-1] in [1, 3, 4]:
                        tensor = tensor.permute(2, 0, 1)
                    return tensor.float() / 255.0
                return torch.tensor(image)


class RandomHorizontalFlip(ImageTransform):
    """
    Randomly flip image horizontally.

    Args:
        p: Probability of flip
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

        if _TORCHVISION_AVAILABLE:
            self._transform = transforms.RandomHorizontalFlip(p=p)
        else:
            self._transform = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            if torch.rand(1).item() < self.p:
                return torch.flip(image, dims=[-1])
            return image


class RandomCrop(ImageTransform):
    """
    Randomly crop image to given size.

    Args:
        size: Target size (height, width) or single int for square
        padding: Optional padding before crop
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        padding: Optional[int] = None,
    ):
        if isinstance(size, int):
            size = (size, size)
        super().__init__(size)
        self.padding = padding

        if _TORCHVISION_AVAILABLE:
            self._transform = transforms.RandomCrop(size, padding=padding)
        else:
            self._transform = None

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self._transform is not None:
            return self._transform(image)
        else:
            h, w = image.shape[-2:]
            th, tw = self.size
            if h < th or w < tw:
                warnings.warn(
                    f"Image size ({h}, {w}) smaller than crop size {self.size}"
                )
                return image
            i = torch.randint(0, h - th + 1, (1,)).item()
            j = torch.randint(0, w - tw + 1, (1,)).item()
            return image[..., i : i + th, j : j + tw]


class ResNet(nn.Module):
    """
    ResNet architecture wrapper.

    Provides access to ResNet-18, 34, 50, 101, 152 with optional pre-trained weights.

    Args:
        num_layers: Number of layers (18, 34, 50, 101, 152)
        pretrained: Whether to load pre-trained ImageNet weights
        num_classes: Number of output classes (default 1000 for ImageNet)
    """

    def __init__(
        self,
        num_layers: int = 50,
        pretrained: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.num_classes = num_classes

        if _TORCHVISION_AVAILABLE:
            model_map = {
                18: (tv_models.resnet18, tv_models.ResNet18_Weights.DEFAULT),
                34: (tv_models.resnet34, tv_models.ResNet34_Weights.DEFAULT),
                50: (tv_models.resnet50, tv_models.ResNet50_Weights.DEFAULT),
                101: (tv_models.resnet101, tv_models.ResNet101_Weights.DEFAULT),
                152: (tv_models.resnet152, tv_models.ResNet152_Weights.DEFAULT),
            }

            if num_layers not in model_map:
                raise ValueError(f"ResNet-{num_layers} not supported. Use 18, 34, 50, 101, or 152")

            model_fn, weights = model_map[num_layers]
            self.model = model_fn(weights=weights if pretrained else None)

            # Modify final layer if num_classes differs
            if num_classes != 1000:
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, num_classes)
        else:
            self.model = None
            warnings.warn(
                "torchvision not available. ResNet will return input unchanged. "
                "Install with: pip install torch9[vision]",
                UserWarning,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            return self.model(x)
        else:
            # Fallback: return input (for testing without torchvision)
            return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class VGG(nn.Module):
    """
    VGG architecture wrapper.

    Provides access to VGG-11, 13, 16, 19 with optional pre-trained weights.

    Args:
        num_layers: Number of layers (11, 13, 16, 19)
        pretrained: Whether to load pre-trained ImageNet weights
        batch_norm: Whether to use batch normalization
        num_classes: Number of output classes
    """

    def __init__(
        self,
        num_layers: int = 16,
        pretrained: bool = False,
        batch_norm: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pretrained = pretrained
        self.batch_norm = batch_norm
        self.num_classes = num_classes

        if _TORCHVISION_AVAILABLE:
            if batch_norm:
                model_map = {
                    11: (tv_models.vgg11_bn, tv_models.VGG11_BN_Weights.DEFAULT),
                    13: (tv_models.vgg13_bn, tv_models.VGG13_BN_Weights.DEFAULT),
                    16: (tv_models.vgg16_bn, tv_models.VGG16_BN_Weights.DEFAULT),
                    19: (tv_models.vgg19_bn, tv_models.VGG19_BN_Weights.DEFAULT),
                }
            else:
                model_map = {
                    11: (tv_models.vgg11, tv_models.VGG11_Weights.DEFAULT),
                    13: (tv_models.vgg13, tv_models.VGG13_Weights.DEFAULT),
                    16: (tv_models.vgg16, tv_models.VGG16_Weights.DEFAULT),
                    19: (tv_models.vgg19, tv_models.VGG19_Weights.DEFAULT),
                }

            if num_layers not in model_map:
                raise ValueError(f"VGG-{num_layers} not supported. Use 11, 13, 16, or 19")

            model_fn, weights = model_map[num_layers]
            self.model = model_fn(weights=weights if pretrained else None)

            # Modify final layer if num_classes differs
            if num_classes != 1000:
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            self.model = None
            warnings.warn(
                "torchvision not available. VGG will return input unchanged. "
                "Install with: pip install torch9[vision]",
                UserWarning,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            return self.model(x)
        else:
            return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class EfficientNet(nn.Module):
    """
    EfficientNet architecture wrapper.

    Provides access to EfficientNet-B0 through B7.

    Args:
        version: EfficientNet version (b0-b7)
        pretrained: Whether to load pre-trained ImageNet weights
        num_classes: Number of output classes
    """

    def __init__(
        self,
        version: str = "b0",
        pretrained: bool = False,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.version = version.lower()
        self.pretrained = pretrained
        self.num_classes = num_classes

        if _TORCHVISION_AVAILABLE:
            model_map = {
                "b0": (tv_models.efficientnet_b0, tv_models.EfficientNet_B0_Weights.DEFAULT),
                "b1": (tv_models.efficientnet_b1, tv_models.EfficientNet_B1_Weights.DEFAULT),
                "b2": (tv_models.efficientnet_b2, tv_models.EfficientNet_B2_Weights.DEFAULT),
                "b3": (tv_models.efficientnet_b3, tv_models.EfficientNet_B3_Weights.DEFAULT),
                "b4": (tv_models.efficientnet_b4, tv_models.EfficientNet_B4_Weights.DEFAULT),
                "b5": (tv_models.efficientnet_b5, tv_models.EfficientNet_B5_Weights.DEFAULT),
                "b6": (tv_models.efficientnet_b6, tv_models.EfficientNet_B6_Weights.DEFAULT),
                "b7": (tv_models.efficientnet_b7, tv_models.EfficientNet_B7_Weights.DEFAULT),
            }

            if self.version not in model_map:
                raise ValueError(f"EfficientNet-{version} not supported. Use b0-b7")

            model_fn, weights = model_map[self.version]
            self.model = model_fn(weights=weights if pretrained else None)

            # Modify final layer if num_classes differs
            if num_classes != 1000:
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            self.model = None
            warnings.warn(
                "torchvision not available. EfficientNet will return input unchanged. "
                "Install with: pip install torch9[vision]",
                UserWarning,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model is not None:
            return self.model(x)
        else:
            return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def get_model(
    name: str,
    pretrained: bool = False,
    num_classes: int = 1000,
    **kwargs,
) -> nn.Module:
    """
    Get a model by name.

    Args:
        name: Model name (e.g., "resnet50", "vgg16", "efficientnet_b0")
        pretrained: Whether to load pre-trained weights
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to the model

    Returns:
        PyTorch model

    Example:
        >>> model = get_model("resnet50", pretrained=True)
        >>> model = get_model("vgg16_bn", pretrained=True, num_classes=10)
    """
    name = name.lower()

    # ResNet variants
    if name.startswith("resnet"):
        num_layers = int(name.replace("resnet", ""))
        return ResNet(num_layers=num_layers, pretrained=pretrained, num_classes=num_classes)

    # VGG variants
    if name.startswith("vgg"):
        bn = "_bn" in name
        num_layers = int(name.replace("vgg", "").replace("_bn", ""))
        return VGG(
            num_layers=num_layers, pretrained=pretrained, batch_norm=bn, num_classes=num_classes
        )

    # EfficientNet variants
    if name.startswith("efficientnet"):
        version = name.replace("efficientnet_", "").replace("efficientnet", "")
        if not version:
            version = "b0"
        return EfficientNet(version=version, pretrained=pretrained, num_classes=num_classes)

    raise ValueError(f"Unknown model: {name}. Use list_models() to see available models.")


def list_models() -> List[str]:
    """
    List all available model names.

    Returns:
        List of model name strings
    """
    return [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "vgg11",
        "vgg13",
        "vgg16",
        "vgg19",
        "vgg11_bn",
        "vgg13_bn",
        "vgg16_bn",
        "vgg19_bn",
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "efficientnet_b4",
        "efficientnet_b5",
        "efficientnet_b6",
        "efficientnet_b7",
    ]


def load_image(
    path: str,
    size: Optional[Tuple[int, int]] = None,
    mode: str = "RGB",
) -> torch.Tensor:
    """
    Load image from path.

    Args:
        path: Path to image file
        size: Target size for resizing (height, width)
        mode: Image mode ("RGB", "L" for grayscale)

    Returns:
        Image tensor of shape (channels, height, width) with values in [0, 1]

    Example:
        >>> image = load_image("photo.jpg")
        >>> image = load_image("photo.jpg", size=(224, 224))
    """
    if _TORCHVISION_AVAILABLE:
        from PIL import Image

        try:
            # Use PIL for more format support
            img = Image.open(path).convert(mode)

            transform_list = [transforms.ToTensor()]
            if size is not None:
                transform_list.insert(0, transforms.Resize(size))

            transform = transforms.Compose(transform_list)
            return transform(img)
        except FileNotFoundError:
            warnings.warn(f"File not found: {path}. Returning placeholder.")
            h, w = size if size else (224, 224)
            channels = 3 if mode == "RGB" else 1
            return torch.zeros(channels, h, w)
    else:
        warnings.warn(
            "torchvision not available. Returning placeholder image. "
            "Install with: pip install torch9[vision]",
            UserWarning,
        )
        h, w = size if size else (224, 224)
        channels = 3 if mode == "RGB" else 1
        return torch.zeros(channels, h, w)


def save_image(
    path: str,
    image: torch.Tensor,
    format: Optional[str] = None,
    quality: int = 95,
) -> None:
    """
    Save image tensor to file.

    Args:
        path: Output path for image
        image: Image tensor of shape (channels, height, width)
        format: Image format ("png", "jpeg"). Inferred from path if None.
        quality: JPEG quality (1-100)

    Example:
        >>> image = torch.rand(3, 224, 224)
        >>> save_image("output.png", image)
    """
    if _TORCHVISION_AVAILABLE:
        # Ensure proper format
        if image.dtype == torch.float32:
            image = (image * 255).clamp(0, 255).to(torch.uint8)

        # Infer format from path
        if format is None:
            format = path.rsplit(".", 1)[-1].lower()

        if format in ["jpg", "jpeg"]:
            write_jpeg(image, path, quality=quality)
        else:
            write_png(image, path)
    else:
        warnings.warn(
            "torchvision not available. Image not saved. "
            "Install with: pip install torch9[vision]",
            UserWarning,
        )
