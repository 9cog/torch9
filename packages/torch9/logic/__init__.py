"""
torch9.logic - Tensor Logic Framework

A comprehensive framework for tensor validation, type checking, and logical operations.
Provides building blocks for type-safe tensor programming with runtime validation.

Features:
- TensorSpec: Define expected tensor shapes, dtypes, and constraints
- TensorGuard: Runtime validation decorators for function inputs/outputs
- TensorOps: Common tensor operations with automatic broadcasting
- DeviceManager: Unified CPU/GPU device management
- MemoryTracker: Track and optimize tensor memory usage

Example:
    >>> from torch9.logic import TensorSpec, tensor_guard, TensorOps
    >>>
    >>> @tensor_guard(input=TensorSpec(shape=(-1, 3, 224, 224), dtype=torch.float32))
    ... def process_images(input):
    ...     return TensorOps.normalize(input)
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import functools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn as nn

__all__ = [
    # Tensor Specification
    "TensorSpec",
    "ShapeSpec",
    # Validation
    "TensorGuard",
    "tensor_guard",
    "validate_tensor",
    "ValidationError",
    # Operations
    "TensorOps",
    "broadcast_tensors",
    "safe_reshape",
    "ensure_batch_dim",
    # Device Management
    "DeviceManager",
    "get_device",
    "to_device",
    "synchronize",
    # Memory
    "MemoryTracker",
    "MemoryStats",
    "memory_stats",
    "clear_cache",
    # Logic Operations
    "TensorLogic",
    "where_masked",
    "logical_combine",
    # Timing
    "Timer",
]


# =============================================================================
# Exceptions
# =============================================================================


class ValidationError(Exception):
    """Raised when tensor validation fails."""

    def __init__(self, message: str, tensor: Optional[torch.Tensor] = None):
        self.tensor = tensor
        super().__init__(message)


# =============================================================================
# Shape Specification
# =============================================================================


@dataclass
class ShapeSpec:
    """
    Specification for tensor shapes with support for dynamic dimensions.

    Args:
        dims: Tuple of dimension sizes. Use -1 for any size, None for optional dim.
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions

    Example:
        >>> spec = ShapeSpec(dims=(-1, 3, 224, 224))  # Batch of 3-channel 224x224 images
        >>> spec = ShapeSpec(min_dims=2, max_dims=4)  # 2D to 4D tensor
    """

    dims: Optional[Tuple[int, ...]] = None
    min_dims: Optional[int] = None
    max_dims: Optional[int] = None
    names: Optional[Tuple[Optional[str], ...]] = None

    def validate(self, shape: torch.Size) -> bool:
        """Check if a shape matches this specification."""
        # Check dimension count
        if self.min_dims is not None and len(shape) < self.min_dims:
            return False
        if self.max_dims is not None and len(shape) > self.max_dims:
            return False

        # Check specific dimensions
        if self.dims is not None:
            if len(shape) != len(self.dims):
                return False
            for actual, expected in zip(shape, self.dims):
                if expected != -1 and expected is not None and actual != expected:
                    return False

        return True

    def describe(self) -> str:
        """Human-readable description of this spec."""
        if self.dims:
            dims_str = ", ".join(
                str(d) if d != -1 else "*" for d in self.dims
            )
            return f"({dims_str})"
        parts = []
        if self.min_dims:
            parts.append(f"min_dims={self.min_dims}")
        if self.max_dims:
            parts.append(f"max_dims={self.max_dims}")
        return f"Shape({', '.join(parts)})" if parts else "Shape(any)"


# =============================================================================
# Tensor Specification
# =============================================================================


@dataclass
class TensorSpec:
    """
    Complete specification for a tensor including shape, dtype, device, and constraints.

    Args:
        shape: Expected shape (use -1 for dynamic dimensions)
        dtype: Expected data type
        device: Expected device ('cpu', 'cuda', or specific like 'cuda:0')
        requires_grad: Whether gradient tracking is required
        contiguous: Whether tensor must be contiguous
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        allow_none: Whether None is acceptable

    Example:
        >>> spec = TensorSpec(
        ...     shape=(-1, 3, 224, 224),
        ...     dtype=torch.float32,
        ...     device='cuda',
        ...     min_value=0.0,
        ...     max_value=1.0
        ... )
        >>> spec.validate(my_tensor)  # Raises ValidationError if invalid
    """

    shape: Optional[Union[Tuple[int, ...], ShapeSpec]] = None
    dtype: Optional[torch.dtype] = None
    device: Optional[Union[str, torch.device]] = None
    requires_grad: Optional[bool] = None
    contiguous: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allow_none: bool = False
    name: str = "tensor"

    def __post_init__(self):
        # Convert shape tuple to ShapeSpec if needed
        if isinstance(self.shape, tuple):
            self.shape = ShapeSpec(dims=self.shape)
        # Normalize device
        if isinstance(self.device, str):
            self.device = torch.device(self.device)

    def validate(self, tensor: Optional[torch.Tensor], raise_error: bool = True) -> bool:
        """
        Validate a tensor against this specification.

        Args:
            tensor: Tensor to validate
            raise_error: If True, raise ValidationError on failure

        Returns:
            True if valid, False otherwise (if raise_error=False)
        """
        errors = []

        # Check None
        if tensor is None:
            if self.allow_none:
                return True
            errors.append(f"{self.name}: Expected tensor, got None")
            if raise_error and errors:
                raise ValidationError("; ".join(errors))
            return False

        # Check type
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"{self.name}: Expected torch.Tensor, got {type(tensor).__name__}")
            if raise_error and errors:
                raise ValidationError("; ".join(errors), tensor)
            return False

        # Check shape
        if self.shape is not None:
            if not self.shape.validate(tensor.shape):
                errors.append(
                    f"{self.name}: Shape {tuple(tensor.shape)} doesn't match {self.shape.describe()}"
                )

        # Check dtype
        if self.dtype is not None and tensor.dtype != self.dtype:
            errors.append(
                f"{self.name}: Expected dtype {self.dtype}, got {tensor.dtype}"
            )

        # Check device
        if self.device is not None:
            if self.device.type != tensor.device.type:
                errors.append(
                    f"{self.name}: Expected device {self.device}, got {tensor.device}"
                )

        # Check requires_grad
        if self.requires_grad is not None and tensor.requires_grad != self.requires_grad:
            errors.append(
                f"{self.name}: Expected requires_grad={self.requires_grad}, "
                f"got {tensor.requires_grad}"
            )

        # Check contiguous
        if self.contiguous and not tensor.is_contiguous():
            errors.append(f"{self.name}: Expected contiguous tensor")

        # Check value range
        if self.min_value is not None:
            min_val = tensor.min().item()
            if min_val < self.min_value:
                errors.append(
                    f"{self.name}: Min value {min_val} < required {self.min_value}"
                )

        if self.max_value is not None:
            max_val = tensor.max().item()
            if max_val > self.max_value:
                errors.append(
                    f"{self.name}: Max value {max_val} > required {self.max_value}"
                )

        if errors:
            if raise_error:
                raise ValidationError("; ".join(errors), tensor)
            return False

        return True

    def coerce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Attempt to coerce a tensor to match this specification.

        Applies dtype conversion, device transfer, and contiguity.
        Does NOT change shape or clamp values.
        """
        if self.dtype is not None and tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)

        if self.device is not None and tensor.device != self.device:
            tensor = tensor.to(self.device)

        if self.contiguous and not tensor.is_contiguous():
            tensor = tensor.contiguous()

        return tensor


# =============================================================================
# Tensor Guard (Decorator)
# =============================================================================


class TensorGuard:
    """
    Decorator for validating function inputs and outputs.

    Example:
        >>> guard = TensorGuard(
        ...     inputs={'x': TensorSpec(shape=(-1, 3), dtype=torch.float32)},
        ...     output=TensorSpec(shape=(-1, 10))
        ... )
        >>> @guard
        ... def model(x):
        ...     return linear(x)
    """

    def __init__(
        self,
        inputs: Optional[Dict[str, TensorSpec]] = None,
        output: Optional[TensorSpec] = None,
        strict: bool = True,
    ):
        self.inputs = inputs or {}
        self.output = output
        self.strict = strict

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate inputs
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for name, spec in self.inputs.items():
                if name in bound.arguments:
                    tensor = bound.arguments[name]
                    spec.name = name
                    spec.validate(tensor, raise_error=self.strict)

            # Call function
            result = func(*args, **kwargs)

            # Validate output
            if self.output is not None:
                self.output.name = "output"
                self.output.validate(result, raise_error=self.strict)

            return result

        return wrapper


def tensor_guard(
    strict: bool = True,
    output: Optional[TensorSpec] = None,
    **input_specs: TensorSpec,
) -> Callable:
    """
    Decorator factory for tensor validation.

    Example:
        >>> @tensor_guard(
        ...     x=TensorSpec(shape=(-1, 3, 224, 224), dtype=torch.float32),
        ...     output=TensorSpec(shape=(-1, 1000))
        ... )
        ... def classify(x):
        ...     return model(x)
    """
    return TensorGuard(inputs=input_specs, output=output, strict=strict)


def validate_tensor(
    tensor: torch.Tensor,
    spec: Optional[TensorSpec] = None,
    **kwargs,
) -> bool:
    """
    Standalone tensor validation function.

    Args:
        tensor: Tensor to validate
        spec: TensorSpec to validate against (or pass kwargs)
        **kwargs: Arguments to create TensorSpec if spec not provided

    Returns:
        True if valid

    Raises:
        ValidationError if invalid
    """
    if spec is None:
        spec = TensorSpec(**kwargs)
    return spec.validate(tensor)


# =============================================================================
# Tensor Operations
# =============================================================================


class TensorOps:
    """
    Collection of common tensor operations with safety checks and broadcasting.
    """

    @staticmethod
    def normalize(
        tensor: torch.Tensor,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        dim: int = -1,
    ) -> torch.Tensor:
        """
        Normalize tensor with optional mean/std or compute from data.

        Args:
            tensor: Input tensor
            mean: Mean values per channel (if None, computed from tensor)
            std: Std values per channel (if None, computed from tensor)
            dim: Dimension to normalize over

        Returns:
            Normalized tensor
        """
        if mean is None:
            mean = tensor.mean(dim=dim, keepdim=True)
        else:
            mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
            # Reshape mean to broadcast correctly
            shape = [1] * tensor.dim()
            shape[dim] = -1
            mean = mean.view(*shape)

        if std is None:
            std = tensor.std(dim=dim, keepdim=True)
        else:
            std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
            shape = [1] * tensor.dim()
            shape[dim] = -1
            std = std.view(*shape)

        return (tensor - mean) / (std + 1e-8)

    @staticmethod
    def safe_divide(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Division with epsilon to avoid division by zero."""
        return numerator / (denominator + eps)

    @staticmethod
    def clamp_norm(
        tensor: torch.Tensor,
        max_norm: float,
        dim: int = -1,
    ) -> torch.Tensor:
        """Clamp tensor norm to maximum value."""
        norm = tensor.norm(dim=dim, keepdim=True)
        scale = torch.clamp(max_norm / (norm + 1e-8), max=1.0)
        return tensor * scale

    @staticmethod
    def softmax_temperature(
        logits: torch.Tensor,
        temperature: float = 1.0,
        dim: int = -1,
    ) -> torch.Tensor:
        """Softmax with temperature scaling."""
        return torch.softmax(logits / temperature, dim=dim)

    @staticmethod
    def gumbel_softmax(
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        dim: int = -1,
    ) -> torch.Tensor:
        """Gumbel-Softmax for differentiable sampling."""
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y_soft = torch.softmax((logits + gumbels) / temperature, dim=dim)

        if hard:
            index = y_soft.argmax(dim=dim, keepdim=True)
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
            return y_hard - y_soft.detach() + y_soft
        return y_soft

    @staticmethod
    def masked_fill_inf(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        neg_inf: bool = True,
    ) -> torch.Tensor:
        """Fill masked positions with infinity (for attention masking)."""
        fill_value = float("-inf") if neg_inf else float("inf")
        return tensor.masked_fill(mask, fill_value)

    @staticmethod
    def gather_nd(
        tensor: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather values from tensor using N-dimensional indices."""
        flat_indices = indices[..., 0]
        for i in range(1, indices.shape[-1]):
            flat_indices = flat_indices * tensor.shape[i] + indices[..., i]
        return tensor.flatten()[flat_indices]

    @staticmethod
    def one_hot(
        indices: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Create one-hot encoding."""
        return torch.zeros(
            *indices.shape, num_classes, dtype=dtype, device=indices.device
        ).scatter_(-1, indices.unsqueeze(-1).long(), 1.0)


def broadcast_tensors(*tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Broadcast tensors to compatible shapes.

    Unlike torch.broadcast_tensors, this handles edge cases gracefully.
    """
    if not tensors:
        return ()

    # Filter out None values
    valid_tensors = [t for t in tensors if t is not None]
    if not valid_tensors:
        return tensors

    return torch.broadcast_tensors(*valid_tensors)


def safe_reshape(
    tensor: torch.Tensor,
    *shape: int,
    allow_copy: bool = True,
) -> torch.Tensor:
    """
    Safely reshape tensor, handling -1 dimensions and contiguity.

    Args:
        tensor: Input tensor
        *shape: Target shape
        allow_copy: If True, make contiguous if needed; else raise error

    Returns:
        Reshaped tensor
    """
    if not tensor.is_contiguous():
        if allow_copy:
            tensor = tensor.contiguous()
        else:
            raise ValueError("Tensor is not contiguous and allow_copy=False")

    return tensor.view(*shape)


def ensure_batch_dim(
    tensor: torch.Tensor,
    expected_dims: int = 4,
) -> Tuple[torch.Tensor, bool]:
    """
    Ensure tensor has batch dimension.

    Args:
        tensor: Input tensor
        expected_dims: Expected number of dimensions with batch

    Returns:
        (tensor with batch dim, was_batched flag)
    """
    was_batched = tensor.dim() == expected_dims
    if not was_batched:
        tensor = tensor.unsqueeze(0)
    return tensor, was_batched


# =============================================================================
# Device Management
# =============================================================================


class DeviceManager:
    """
    Unified device management for tensors and models.

    Example:
        >>> dm = DeviceManager(prefer_cuda=True)
        >>> tensor = dm.to_device(my_tensor)
        >>> model = dm.to_device(my_model)
    """

    _default_device: Optional[torch.device] = None

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        prefer_cuda: bool = True,
    ):
        if device is not None:
            self.device = torch.device(device)
        elif prefer_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @classmethod
    def set_default(cls, device: Union[str, torch.device]) -> None:
        """Set default device for all operations."""
        cls._default_device = torch.device(device)

    @classmethod
    def get_default(cls) -> torch.device:
        """Get default device."""
        if cls._default_device is not None:
            return cls._default_device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(
        self,
        obj: Union[torch.Tensor, nn.Module],
        non_blocking: bool = True,
    ) -> Union[torch.Tensor, nn.Module]:
        """Move tensor or module to managed device."""
        return obj.to(self.device, non_blocking=non_blocking)

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if using MPS (Apple Silicon) device."""
        return self.device.type == "mps"

    def synchronize(self) -> None:
        """Synchronize device (wait for all operations to complete)."""
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def device_context(self):
        """Context manager for device operations."""
        old_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        try:
            if self.is_cuda:
                torch.cuda.set_device(self.device)
            yield self.device
        finally:
            if old_device is not None:
                torch.cuda.set_device(old_device)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(
    obj: Union[torch.Tensor, nn.Module],
    device: Optional[Union[str, torch.device]] = None,
    non_blocking: bool = True,
) -> Union[torch.Tensor, nn.Module]:
    """Move tensor or module to device."""
    if device is None:
        device = get_device()
    return obj.to(device, non_blocking=non_blocking)


def synchronize(device: Optional[torch.device] = None) -> None:
    """Synchronize device operations."""
    if device is None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================
# Memory Management
# =============================================================================


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    allocated: int
    reserved: int
    max_allocated: int
    device: torch.device

    def __repr__(self) -> str:
        return (
            f"MemoryStats(device={self.device}, "
            f"allocated={self.allocated / 1e6:.1f}MB, "
            f"reserved={self.reserved / 1e6:.1f}MB, "
            f"max_allocated={self.max_allocated / 1e6:.1f}MB)"
        )


class MemoryTracker:
    """
    Track and manage GPU memory usage.

    Example:
        >>> tracker = MemoryTracker()
        >>> with tracker.track("forward_pass"):
        ...     output = model(input)
        >>> print(tracker.report())
    """

    def __init__(self):
        self.checkpoints: Dict[str, MemoryStats] = {}
        self._stack: List[Tuple[str, int]] = []

    def snapshot(self, name: str = "snapshot") -> MemoryStats:
        """Take memory snapshot."""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0, torch.device("cpu"))

        stats = MemoryStats(
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            max_allocated=torch.cuda.max_memory_allocated(),
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self.checkpoints[name] = stats
        return stats

    @contextmanager
    def track(self, name: str):
        """Context manager to track memory for a code block."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()
            self._stack.append((name, start_mem))

        yield

        if torch.cuda.is_available() and self._stack:
            name, start_mem = self._stack.pop()
            self.checkpoints[name] = MemoryStats(
                allocated=torch.cuda.memory_allocated() - start_mem,
                reserved=torch.cuda.memory_reserved(),
                max_allocated=torch.cuda.max_memory_allocated(),
                device=torch.device("cuda", torch.cuda.current_device()),
            )

    def report(self) -> str:
        """Generate memory usage report."""
        lines = ["Memory Usage Report", "=" * 40]
        for name, stats in self.checkpoints.items():
            lines.append(f"{name}: {stats.allocated / 1e6:.2f}MB allocated")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self._stack.clear()


def memory_stats(device: Optional[torch.device] = None) -> MemoryStats:
    """Get current memory statistics."""
    if not torch.cuda.is_available():
        return MemoryStats(0, 0, 0, torch.device("cpu"))

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())

    return MemoryStats(
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
        max_allocated=torch.cuda.max_memory_allocated(device),
        device=device,
    )


def clear_cache() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Tensor Logic Operations
# =============================================================================


class TensorLogic:
    """
    Logical operations on tensors with proper broadcasting and type handling.
    """

    @staticmethod
    def all_equal(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-6) -> bool:
        """Check if two tensors are approximately equal."""
        if a.shape != b.shape:
            return False
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        return torch.allclose(a, b, atol=tol)

    @staticmethod
    def any_nan(tensor: torch.Tensor) -> bool:
        """Check if tensor contains any NaN values."""
        return torch.isnan(tensor).any().item()

    @staticmethod
    def any_inf(tensor: torch.Tensor) -> bool:
        """Check if tensor contains any infinite values."""
        return torch.isinf(tensor).any().item()

    @staticmethod
    def is_valid(tensor: torch.Tensor) -> bool:
        """Check if tensor contains only valid (finite) values."""
        return torch.isfinite(tensor).all().item()

    @staticmethod
    def where_masked(
        condition: torch.Tensor,
        true_tensor: torch.Tensor,
        false_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Enhanced torch.where with automatic broadcasting."""
        condition = condition.bool()
        return torch.where(condition, true_tensor, false_tensor)

    @staticmethod
    def logical_combine(
        tensors: List[torch.Tensor],
        op: str = "and",
    ) -> torch.Tensor:
        """
        Combine multiple boolean tensors with logical operation.

        Args:
            tensors: List of boolean tensors
            op: Operation ('and', 'or', 'xor')

        Returns:
            Combined boolean tensor
        """
        if not tensors:
            raise ValueError("Need at least one tensor")

        result = tensors[0].bool()
        for t in tensors[1:]:
            t = t.bool()
            if op == "and":
                result = result & t
            elif op == "or":
                result = result | t
            elif op == "xor":
                result = result ^ t
            else:
                raise ValueError(f"Unknown operation: {op}")

        return result

    @staticmethod
    def count_nonzero(
        tensor: torch.Tensor,
        dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Count non-zero elements."""
        return torch.count_nonzero(tensor, dim=dim)

    @staticmethod
    def argmax_2d(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get row and column indices of maximum values in 2D tensor."""
        flat_idx = tensor.view(-1).argmax()
        row = flat_idx // tensor.shape[1]
        col = flat_idx % tensor.shape[1]
        return row, col

    @staticmethod
    def top_k_mask(
        tensor: torch.Tensor,
        k: int,
        dim: int = -1,
    ) -> torch.Tensor:
        """Create mask for top-k values along dimension."""
        _, indices = torch.topk(tensor, k, dim=dim)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(dim, indices, True)
        return mask


# Convenience function exports
def where_masked(
    condition: torch.Tensor,
    true_tensor: torch.Tensor,
    false_tensor: torch.Tensor,
) -> torch.Tensor:
    """Enhanced torch.where with automatic broadcasting."""
    return TensorLogic.where_masked(condition, true_tensor, false_tensor)


def logical_combine(
    tensors: List[torch.Tensor],
    op: str = "and",
) -> torch.Tensor:
    """Combine multiple boolean tensors with logical operation."""
    return TensorLogic.logical_combine(tensors, op)


# =============================================================================
# Timer
# =============================================================================


class Timer:
    """
    Simple timer for measuring execution time.

    Example:
        >>> timer = Timer().start()
        >>> # ... do something ...
        >>> elapsed = timer.stop()

        >>> with Timer().measure() as timer:
        ...     # ... do something ...
        >>> print(timer.elapsed)
    """

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._laps: List[float] = []

    def start(self) -> "Timer":
        """Start the timer."""
        import time
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        import time
        self._end = time.perf_counter()
        return self.elapsed

    def lap(self) -> float:
        """Record a lap time."""
        import time
        lap_time = time.perf_counter() - (self._start or 0)
        self._laps.append(lap_time)
        return lap_time

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
        import time
        if self._start is None:
            return 0.0
        end = self._end or time.perf_counter()
        return end - self._start

    @contextmanager
    def measure(self):
        """Context manager for timing a block."""
        self.start()
        try:
            yield self
        finally:
            self.stop()
