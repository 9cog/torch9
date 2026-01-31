"""
torch9.luatorch - Unified LuaTorch Core

A Python bridge to the core abstractions of the Torch7/LuaTorch framework,
implemented on top of modern PyTorch. Provides Torch7-compatible APIs for
tensors, neural network modules, loss functions, optimizers, serialization,
and utilities.

This module faithfully reproduces Torch7's programming model:
- Typed tensor constructors (FloatTensor, DoubleTensor, etc.)
- Module with explicit updateOutput/updateGradInput/accGradParameters
- Functional optimizers that take (opfunc, x, config, state)
- Class registry with typename/isTypeOf introspection
- tic/toc timing utilities

Example:
    >>> from torch9.luatorch import nn, optim, FloatTensor
    >>>
    >>> model = nn.Sequential()
    >>> model.add(nn.Linear(784, 128))
    >>> model.add(nn.ReLU())
    >>> model.add(nn.Linear(128, 10))
    >>> model.add(nn.LogSoftMax())
    >>>
    >>> criterion = nn.ClassNLLCriterion()
    >>> x = FloatTensor(32, 784).normal_()
    >>> output = model.forward(x)
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
import copy
import io
import math
import os
import time
import warnings
from collections import OrderedDict

import torch

__all__ = [
    # Tensor types
    "FloatTensor",
    "DoubleTensor",
    "HalfTensor",
    "ByteTensor",
    "CharTensor",
    "ShortTensor",
    "IntTensor",
    "LongTensor",
    "Storage",
    "setdefaulttensortype",
    "getdefaulttensortype",
    # Class system
    "ClassRegistry",
    "class_registry",
    "typename",
    "isTypeOf",
    # Neural network module system
    "nn",
    # Functional optimizers
    "optim",
    # Serialization
    "save",
    "load",
    # Utilities
    "Timer",
    "tic",
    "toc",
    "paths",
]


# =============================================================================
# Class Registry - Torch7 class system
# =============================================================================


class ClassRegistry:
    """Torch7-style class registration and introspection.

    Provides typename() and isTypeOf() for runtime type checking,
    mirroring torch.class() / torch.typename() / torch.isTypeOf() from Lua.
    """

    def __init__(self):
        self._registry: Dict[str, Type] = {}
        self._names: Dict[Type, str] = {}

    def register(self, name: str, cls: Type) -> None:
        """Register a class under a canonical Torch7 name."""
        self._registry[name] = cls
        self._names[cls] = name

    def get(self, name: str) -> Optional[Type]:
        """Look up a class by its Torch7 name."""
        return self._registry.get(name)

    def typename(self, obj: Any) -> str:
        """Return the registered Torch7 name, or the Python class name."""
        cls = type(obj)
        for klass in cls.__mro__:
            if klass in self._names:
                return self._names[klass]
        return cls.__qualname__

    def isTypeOf(self, obj: Any, name_or_type: Union[str, Type]) -> bool:
        """Check whether *obj* is an instance of the named type."""
        if isinstance(name_or_type, str):
            target = self._registry.get(name_or_type)
            if target is None:
                return False
            return isinstance(obj, target)
        return isinstance(obj, name_or_type)

    @property
    def registered_names(self) -> List[str]:
        return list(self._registry.keys())


class_registry = ClassRegistry()


def typename(obj: Any) -> str:
    """Return the Torch7 type name of *obj*."""
    return class_registry.typename(obj)


def isTypeOf(obj: Any, name_or_type: Union[str, Type]) -> bool:
    """Check whether *obj* is an instance of the given type."""
    return class_registry.isTypeOf(obj, name_or_type)


# =============================================================================
# Typed Tensor Constructors - Torch7 tensor type system
# =============================================================================

_DEFAULT_TENSOR_TYPE = "torch.FloatTensor"

_DTYPE_MAP = {
    "torch.FloatTensor": torch.float32,
    "torch.DoubleTensor": torch.float64,
    "torch.HalfTensor": torch.float16,
    "torch.ByteTensor": torch.uint8,
    "torch.CharTensor": torch.int8,
    "torch.ShortTensor": torch.int16,
    "torch.IntTensor": torch.int32,
    "torch.LongTensor": torch.int64,
}


def setdefaulttensortype(typename: str) -> None:
    """Set the default tensor type (e.g. 'torch.FloatTensor')."""
    global _DEFAULT_TENSOR_TYPE
    if typename not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown tensor type '{typename}'. "
            f"Valid types: {list(_DTYPE_MAP.keys())}"
        )
    _DEFAULT_TENSOR_TYPE = typename


def getdefaulttensortype() -> str:
    """Return the current default tensor type string."""
    return _DEFAULT_TENSOR_TYPE


def _make_tensor(dtype: torch.dtype, *args) -> torch.Tensor:
    """Create a tensor of the given dtype from flexible arguments.

    Supports:
        Tensor()            -> empty 0-d tensor
        Tensor(5)           -> 1-d tensor of size 5
        Tensor(3, 4)        -> 2-d tensor of size 3x4
        Tensor(sizes...)    -> n-d tensor
        Tensor(existing)    -> copy of existing tensor, cast to dtype
    """
    if len(args) == 0:
        return torch.empty(0, dtype=dtype)
    if len(args) == 1:
        arg = args[0]
        if isinstance(arg, torch.Tensor):
            return arg.to(dtype=dtype).clone()
        if isinstance(arg, (list, tuple)):
            return torch.tensor(arg, dtype=dtype)
        # Single int -> 1-d
        return torch.empty(int(arg), dtype=dtype)
    # Multiple ints -> shape
    shape = tuple(int(a) for a in args)
    return torch.empty(shape, dtype=dtype)


def FloatTensor(*args) -> torch.Tensor:
    """Create a Float (32-bit) tensor. Mirrors torch.FloatTensor from Lua."""
    return _make_tensor(torch.float32, *args)


def DoubleTensor(*args) -> torch.Tensor:
    """Create a Double (64-bit) tensor. Mirrors torch.DoubleTensor from Lua."""
    return _make_tensor(torch.float64, *args)


def HalfTensor(*args) -> torch.Tensor:
    """Create a Half (16-bit float) tensor."""
    return _make_tensor(torch.float16, *args)


def ByteTensor(*args) -> torch.Tensor:
    """Create a Byte (uint8) tensor."""
    return _make_tensor(torch.uint8, *args)


def CharTensor(*args) -> torch.Tensor:
    """Create a Char (int8) tensor."""
    return _make_tensor(torch.int8, *args)


def ShortTensor(*args) -> torch.Tensor:
    """Create a Short (int16) tensor."""
    return _make_tensor(torch.int16, *args)


def IntTensor(*args) -> torch.Tensor:
    """Create an Int (int32) tensor."""
    return _make_tensor(torch.int32, *args)


def LongTensor(*args) -> torch.Tensor:
    """Create a Long (int64) tensor."""
    return _make_tensor(torch.int64, *args)


# =============================================================================
# Storage - Torch7 storage abstraction
# =============================================================================


class Storage:
    """Thin wrapper around torch.Storage, providing Torch7-style access.

    In Torch7, Storage is the raw data buffer that tensors reference.
    Multiple tensors can share the same Storage with different offsets/strides.
    """

    def __init__(self, size_or_data: Union[int, list, torch.Tensor] = 0,
                 dtype: torch.dtype = torch.float32):
        if isinstance(size_or_data, torch.Tensor):
            self._tensor = size_or_data.detach().flatten().clone()
        elif isinstance(size_or_data, (list, tuple)):
            self._tensor = torch.tensor(size_or_data, dtype=dtype)
        else:
            self._tensor = torch.empty(int(size_or_data), dtype=dtype)

    def size(self) -> int:
        return self._tensor.numel()

    def __len__(self) -> int:
        return self.size()

    def __getitem__(self, idx: int) -> float:
        return self._tensor[idx].item()

    def __setitem__(self, idx: int, value: float) -> None:
        self._tensor[idx] = value

    def fill(self, value: float) -> "Storage":
        self._tensor.fill_(value)
        return self

    def copy(self, other: "Storage") -> "Storage":
        self._tensor.copy_(other._tensor)
        return self

    def resize(self, size: int) -> "Storage":
        if size > self.size():
            extra = torch.empty(size - self.size(), dtype=self._tensor.dtype)
            self._tensor = torch.cat([self._tensor, extra])
        else:
            self._tensor = self._tensor[:size]
        return self

    def tolist(self) -> list:
        return self._tensor.tolist()

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    def __repr__(self) -> str:
        return f"Storage(size={self.size()}, dtype={self.dtype})"


# =============================================================================
# Neural Network Module System - nn.*
# =============================================================================


class _Module:
    """Base class mirroring Torch7's nn.Module.

    Key differences from PyTorch's nn.Module:
    - Explicit updateOutput/updateGradInput/accGradParameters methods
    - forward() delegates to updateOutput(), backward() orchestrates the chain
    - output and gradInput are stored as instance attributes
    - parameters() returns (params_list, grad_params_list) tuple
    """

    def __init__(self):
        self.output: Optional[torch.Tensor] = None
        self.gradInput: Optional[torch.Tensor] = None
        self.train_mode: bool = True

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        """Compute the output from *input*. Subclasses must override."""
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient w.r.t. input. Subclasses should override."""
        return self.gradInput

    def accGradParameters(
        self, input: torch.Tensor, gradOutput: torch.Tensor, scale: float = 1.0
    ) -> None:
        """Accumulate parameter gradients. Override if module has params."""
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Run forward pass (delegates to updateOutput)."""
        return self.updateOutput(input)

    def backward(
        self,
        input: torch.Tensor,
        gradOutput: torch.Tensor,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Run backward pass: gradient w.r.t. input + accumulate param grads."""
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput, scale)
        return self.gradInput

    def parameters(self) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Return (params, gradParams) lists, or None if no parameters."""
        return None

    def zeroGradParameters(self) -> None:
        """Zero all parameter gradients."""
        result = self.parameters()
        if result is not None:
            _, grad_params = result
            for gp in grad_params:
                gp.zero_()

    def updateParameters(self, learning_rate: float) -> None:
        """Simple SGD update: param -= lr * grad."""
        result = self.parameters()
        if result is not None:
            params, grad_params = result
            for p, gp in zip(params, grad_params):
                p.add_(gp, alpha=-learning_rate)

    def training(self) -> "_Module":
        """Set training mode (enables dropout, BN stats)."""
        self.train_mode = True
        return self

    def evaluate(self) -> "_Module":
        """Set evaluation mode (disables dropout, uses running stats)."""
        self.train_mode = False
        return self

    def type(self, dtype: torch.dtype) -> "_Module":
        """Cast all parameters and buffers to *dtype*."""
        result = self.parameters()
        if result is not None:
            params, grad_params = result
            for i, p in enumerate(params):
                params[i] = p.to(dtype)
            for i, gp in enumerate(grad_params):
                grad_params[i] = gp.to(dtype)
        return self

    def float(self) -> "_Module":
        return self.type(torch.float32)

    def double(self) -> "_Module":
        return self.type(torch.float64)

    def clone(self) -> "_Module":
        """Deep copy the module."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"{typename(self)}()"


class _Container(_Module):
    """Base container holding a list of child modules.

    Mirrors Torch7's nn.Container with add/get/size and recursive
    parameter collection.
    """

    def __init__(self):
        super().__init__()
        self.modules: List[_Module] = []

    def add(self, module: _Module) -> "_Container":
        """Append a child module. Returns self for chaining."""
        self.modules.append(module)
        return self

    def get(self, index: int) -> _Module:
        """Get the child module at *index* (0-based)."""
        return self.modules[index]

    def size(self) -> int:
        """Number of child modules."""
        return len(self.modules)

    def parameters(self) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """Recursively collect parameters from all children."""
        all_params: List[torch.Tensor] = []
        all_grads: List[torch.Tensor] = []
        for m in self.modules:
            result = m.parameters()
            if result is not None:
                params, grads = result
                all_params.extend(params)
                all_grads.extend(grads)
        if all_params:
            return all_params, all_grads
        return None

    def zeroGradParameters(self) -> None:
        for m in self.modules:
            m.zeroGradParameters()

    def training(self) -> "_Container":
        super().training()
        for m in self.modules:
            m.training()
        return self

    def evaluate(self) -> "_Container":
        super().evaluate()
        for m in self.modules:
            m.evaluate()
        return self

    def type(self, dtype: torch.dtype) -> "_Container":
        for m in self.modules:
            m.type(dtype)
        return self

    def __repr__(self) -> str:
        lines = [f"{typename(self)} {{"]
        for i, m in enumerate(self.modules):
            lines.append(f"  ({i}): {m}")
        lines.append("}")
        return "\n".join(lines)


class _Sequential(_Container):
    """Linear composition: input -> m1 -> m2 -> ... -> output.

    Mirrors nn.Sequential from Torch7 exactly.
    """

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        current = input
        for module in self.modules:
            current = module.updateOutput(current)
        self.output = current
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        current_grad = gradOutput
        # Walk backward through modules
        for i in range(len(self.modules) - 1, 0, -1):
            module = self.modules[i]
            prev_output = self.modules[i - 1].output
            current_grad = module.updateGradInput(prev_output, current_grad)
        if self.modules:
            self.modules[0].updateGradInput(input, current_grad)
            self.gradInput = self.modules[0].gradInput
        else:
            self.gradInput = gradOutput
        return self.gradInput

    def accGradParameters(
        self, input: torch.Tensor, gradOutput: torch.Tensor, scale: float = 1.0
    ) -> None:
        current_grad = gradOutput
        for i in range(len(self.modules) - 1, 0, -1):
            prev_output = self.modules[i - 1].output
            self.modules[i].accGradParameters(prev_output, current_grad, scale)
            current_grad = self.modules[i].gradInput
        if self.modules:
            self.modules[0].accGradParameters(input, current_grad, scale)

    def insert(self, module: _Module, index: int) -> "_Sequential":
        """Insert a module at position *index*."""
        self.modules.insert(index, module)
        return self

    def remove(self, index: int) -> _Module:
        """Remove and return the module at *index*."""
        return self.modules.pop(index)


class _ConcatTable(_Container):
    """Applies each child to the same input, returns list of outputs.

    Mirrors nn.ConcatTable: input -> [m1(input), m2(input), ...].
    """

    def updateOutput(self, input: torch.Tensor) -> List[torch.Tensor]:
        self.output = [m.updateOutput(input) for m in self.modules]
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: List[torch.Tensor]
    ) -> torch.Tensor:
        self.gradInput = torch.zeros_like(input)
        for m, go in zip(self.modules, gradOutput):
            gi = m.updateGradInput(input, go)
            if gi is not None:
                self.gradInput.add_(gi)
        return self.gradInput


class _Parallel(_Container):
    """Applies each child to the corresponding input split along a dimension.

    Mirrors nn.Parallel: splits input along input_dim, applies modules,
    concatenates along output_dim.
    """

    def __init__(self, input_dim: int = 0, output_dim: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        splits = torch.chunk(input, len(self.modules), dim=self.input_dim)
        outputs = []
        for module, chunk in zip(self.modules, splits):
            outputs.append(module.updateOutput(chunk))
        self.output = torch.cat(outputs, dim=self.output_dim)
        return self.output


# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------


class _Linear(_Module):
    """Fully connected layer: output = input @ weight^T + bias.

    Matches Torch7's nn.Linear(inputSize, outputSize, [bias]).
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.empty(output_size, input_size)
        self.gradWeight = torch.zeros(output_size, input_size)
        stdv = 1.0 / math.sqrt(input_size)
        self.weight.uniform_(-stdv, stdv)
        self.has_bias = bias
        if bias:
            self.bias = torch.empty(output_size).uniform_(-stdv, stdv)
            self.gradBias = torch.zeros(output_size)
        else:
            self.bias = None
            self.gradBias = None
        self._input: Optional[torch.Tensor] = None

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self._input = input
        self.output = input @ self.weight.t()
        if self.has_bias:
            self.output = self.output + self.bias
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput @ self.weight
        return self.gradInput

    def accGradParameters(
        self, input: torch.Tensor, gradOutput: torch.Tensor, scale: float = 1.0
    ) -> None:
        if gradOutput.dim() == 1:
            self.gradWeight.add_(
                torch.outer(gradOutput, input), alpha=scale
            )
        else:
            self.gradWeight.add_(gradOutput.t() @ input, alpha=scale)
        if self.has_bias:
            if gradOutput.dim() == 1:
                self.gradBias.add_(gradOutput, alpha=scale)
            else:
                self.gradBias.add_(gradOutput.sum(0), alpha=scale)

    def parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.has_bias:
            return [self.weight, self.bias], [self.gradWeight, self.gradBias]
        return [self.weight], [self.gradWeight]

    def __repr__(self) -> str:
        return (
            f"nn.Linear({self.input_size} -> {self.output_size}"
            f"{', bias=False' if not self.has_bias else ''})"
        )


class _SpatialConvolution(_Module):
    """2D convolution layer using PyTorch's F.conv2d as backend.

    Mirrors nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH).
    """

    def __init__(
        self,
        n_input_plane: int,
        n_output_plane: int,
        kw: int,
        kh: int,
        dw: int = 1,
        dh: int = 1,
        pad_w: int = 0,
        pad_h: int = 0,
    ):
        super().__init__()
        self.n_input_plane = n_input_plane
        self.n_output_plane = n_output_plane
        self.kw, self.kh = kw, kh
        self.dw, self.dh = dw, dh
        self.pad_w, self.pad_h = pad_w, pad_h
        stdv = 1.0 / math.sqrt(n_input_plane * kw * kh)
        self.weight = torch.empty(
            n_output_plane, n_input_plane, kh, kw
        ).uniform_(-stdv, stdv)
        self.bias = torch.empty(n_output_plane).uniform_(-stdv, stdv)
        self.gradWeight = torch.zeros_like(self.weight)
        self.gradBias = torch.zeros_like(self.bias)

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self._input = input
        self.output = torch.nn.functional.conv2d(
            input,
            self.weight,
            self.bias,
            stride=(self.dh, self.dw),
            padding=(self.pad_h, self.pad_w),
        )
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = torch.nn.functional.conv_transpose2d(
            gradOutput,
            self.weight,
            stride=(self.dh, self.dw),
            padding=(self.pad_h, self.pad_w),
        )
        return self.gradInput

    def accGradParameters(
        self, input: torch.Tensor, gradOutput: torch.Tensor, scale: float = 1.0
    ) -> None:
        # Use unfold-based gradient accumulation
        batch = input.shape[0]
        # gradBias
        self.gradBias.add_(
            gradOutput.sum(dim=(0, 2, 3)), alpha=scale
        )
        # gradWeight via grouped conv trick
        inp_unfold = torch.nn.functional.unfold(
            input,
            kernel_size=(self.kh, self.kw),
            stride=(self.dh, self.dw),
            padding=(self.pad_h, self.pad_w),
        )
        grad_flat = gradOutput.view(batch, self.n_output_plane, -1)
        gw = torch.bmm(grad_flat, inp_unfold.transpose(1, 2))
        self.gradWeight.add_(
            gw.sum(0).view_as(self.gradWeight), alpha=scale
        )

    def parameters(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return [self.weight, self.bias], [self.gradWeight, self.gradBias]

    def __repr__(self) -> str:
        return (
            f"nn.SpatialConvolution("
            f"{self.n_input_plane} -> {self.n_output_plane}, "
            f"{self.kw}x{self.kh}, "
            f"stride={self.dw}x{self.dh}, "
            f"pad={self.pad_w}x{self.pad_h})"
        )


class _SpatialBatchNormalization(_Module):
    """Batch normalization for 4-D (N, C, H, W) tensors.

    Mirrors nn.SpatialBatchNormalization(nOutput, eps, momentum, affine).
    """

    def __init__(
        self,
        n_output: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.n_output = n_output
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.running_mean = torch.zeros(n_output)
        self.running_var = torch.ones(n_output)
        if affine:
            self.weight = torch.ones(n_output)
            self.bias = torch.zeros(n_output)
            self.gradWeight = torch.zeros(n_output)
            self.gradBias = torch.zeros(n_output)
        else:
            self.weight = None
            self.bias = None
            self.gradWeight = None
            self.gradBias = None

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        if self.train_mode:
            mean = input.mean(dim=(0, 2, 3))
            var = input.var(dim=(0, 2, 3), unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        self._mean = mean
        self._var = var
        shape = (1, self.n_output, 1, 1)
        normed = (input - mean.view(shape)) / torch.sqrt(var.view(shape) + self.eps)
        self._normed = normed
        if self.affine:
            self.output = normed * self.weight.view(shape) + self.bias.view(shape)
        else:
            self.output = normed
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        shape = (1, self.n_output, 1, 1)
        std_inv = 1.0 / torch.sqrt(self._var.view(shape) + self.eps)
        if self.affine:
            gradOutput = gradOutput * self.weight.view(shape)
        n = input.shape[0] * input.shape[2] * input.shape[3]
        self.gradInput = (
            std_inv / n * (
                n * gradOutput
                - gradOutput.sum(dim=(0, 2, 3)).view(shape)
                - self._normed * (gradOutput * self._normed).sum(dim=(0, 2, 3)).view(shape)
            )
        )
        return self.gradInput

    def accGradParameters(
        self, input: torch.Tensor, gradOutput: torch.Tensor, scale: float = 1.0
    ) -> None:
        if self.affine:
            self.gradWeight.add_(
                (gradOutput * self._normed).sum(dim=(0, 2, 3)), alpha=scale
            )
            self.gradBias.add_(
                gradOutput.sum(dim=(0, 2, 3)), alpha=scale
            )

    def parameters(self) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if self.affine:
            return [self.weight, self.bias], [self.gradWeight, self.gradBias]
        return None

    def __repr__(self) -> str:
        return f"nn.SpatialBatchNormalization({self.n_output}, eps={self.eps})"


class _Dropout(_Module):
    """Dropout regularization.

    Mirrors nn.Dropout(p) - zeros elements with probability p during training.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask: Optional[torch.Tensor] = None

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        if self.train_mode and self.p > 0:
            self._mask = torch.bernoulli(
                torch.full_like(input, 1.0 - self.p)
            ) / (1.0 - self.p)
            self.output = input * self._mask
        else:
            self.output = input
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        if self.train_mode and self._mask is not None:
            self.gradInput = gradOutput * self._mask
        else:
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self) -> str:
        return f"nn.Dropout({self.p})"


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


class _ReLU(_Module):
    """Rectified Linear Unit: max(0, x)."""

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.relu(input)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput * (input > 0).float()
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.ReLU()"


class _Tanh(_Module):
    """Hyperbolic tangent activation."""

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.tanh(input)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput * (1.0 - self.output ** 2)
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.Tanh()"


class _Sigmoid(_Module):
    """Logistic sigmoid activation."""

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.sigmoid(input)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput * self.output * (1.0 - self.output)
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.Sigmoid()"


class _LogSoftMax(_Module):
    """Log-softmax: log(softmax(x)), applied along dim=1."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.log_softmax(input, dim=self.dim)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        softmax = torch.exp(self.output)
        self.gradInput = gradOutput - softmax * gradOutput.sum(
            dim=self.dim, keepdim=True
        )
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.LogSoftMax()"


class _SoftMax(_Module):
    """Softmax activation, applied along dim=1."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.softmax(input, dim=self.dim)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        dot = (gradOutput * self.output).sum(dim=self.dim, keepdim=True)
        self.gradInput = self.output * (gradOutput - dot)
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.SoftMax()"


class _LeakyReLU(_Module):
    """Leaky ReLU: max(negative_slope * x, x)."""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.where(input >= 0, input, input * self.negative_slope)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput * torch.where(
            input >= 0,
            torch.ones_like(input),
            torch.full_like(input, self.negative_slope),
        )
        return self.gradInput

    def __repr__(self) -> str:
        return f"nn.LeakyReLU({self.negative_slope})"


class _ELU(_Module):
    """Exponential Linear Unit: x if x > 0, else alpha * (exp(x) - 1)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = torch.where(
            input > 0, input, self.alpha * (torch.exp(input) - 1)
        )
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput * torch.where(
            input > 0, torch.ones_like(input), self.output + self.alpha
        )
        return self.gradInput

    def __repr__(self) -> str:
        return f"nn.ELU({self.alpha})"


# ---------------------------------------------------------------------------
# Reshape / View utilities
# ---------------------------------------------------------------------------


class _View(_Module):
    """Reshape the input tensor. Mirrors nn.View(sizes...)."""

    def __init__(self, *sizes: int):
        super().__init__()
        self.sizes = sizes

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = input.contiguous().view(*self.sizes)
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput.contiguous().view_as(input)
        return self.gradInput

    def __repr__(self) -> str:
        return f"nn.View({', '.join(str(s) for s in self.sizes)})"


class _Identity(_Module):
    """Pass input through unchanged. Mirrors nn.Identity()."""

    def updateOutput(self, input: torch.Tensor) -> torch.Tensor:
        self.output = input
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, gradOutput: torch.Tensor
    ) -> torch.Tensor:
        self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self) -> str:
        return "nn.Identity()"


# =============================================================================
# Criterion (Loss Functions)
# =============================================================================


class _Criterion:
    """Base class for all loss functions, mirroring Torch7's nn.Criterion.

    Has the same forward/backward contract as Module, but takes
    (input, target) instead of just (input).
    """

    def __init__(self):
        self.output: float = 0.0
        self.gradInput: Optional[torch.Tensor] = None

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.gradInput

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        return self.updateOutput(input, target)

    def backward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self.updateGradInput(input, target)

    def __repr__(self) -> str:
        return f"{typename(self)}()"


class _MSECriterion(_Criterion):
    """Mean Squared Error: mean((input - target)^2)."""

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.mse_loss(input, target).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = input.numel()
        self.gradInput = 2.0 * (input - target) / n
        return self.gradInput


class _ClassNLLCriterion(_Criterion):
    """Negative log-likelihood loss for classification.

    Expects input = log-probabilities (N, C), target = class indices (N,).
    """

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.nll_loss(input, target).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = input.shape[0]
        self.gradInput = torch.zeros_like(input)
        self.gradInput.scatter_(1, target.unsqueeze(1), -1.0 / n)
        return self.gradInput


class _CrossEntropyCriterion(_Criterion):
    """Cross-entropy loss (combines LogSoftMax + ClassNLLCriterion)."""

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.cross_entropy(input, target).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = input.shape[0]
        probs = torch.softmax(input, dim=1)
        self.gradInput = probs.clone()
        self.gradInput.scatter_add_(
            1, target.unsqueeze(1), torch.full((n, 1), -1.0)
        )
        self.gradInput /= n
        return self.gradInput


class _BCECriterion(_Criterion):
    """Binary Cross-Entropy loss. Input and target are in [0, 1]."""

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.binary_cross_entropy(
            input, target
        ).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        eps = 1e-12
        n = input.numel()
        self.gradInput = (
            (-target / (input + eps) + (1 - target) / (1 - input + eps)) / n
        )
        return self.gradInput


class _AbsCriterion(_Criterion):
    """Mean Absolute Error: mean(|input - target|)."""

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.l1_loss(input, target).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = input.numel()
        self.gradInput = (input - target).sign() / n
        return self.gradInput


class _SmoothL1Criterion(_Criterion):
    """Smooth L1 / Huber loss."""

    def updateOutput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> float:
        self.output = torch.nn.functional.smooth_l1_loss(input, target).item()
        return self.output

    def updateGradInput(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        n = input.numel()
        diff = input - target
        self.gradInput = torch.where(
            diff.abs() < 1, diff / n, diff.sign() / n
        )
        return self.gradInput


# =============================================================================
# nn namespace - collects all neural network components
# =============================================================================


class _NNNamespace:
    """Namespace object that mirrors ``require 'nn'`` from Torch7.

    Access layers as nn.Linear, nn.Sequential, nn.ReLU, etc.
    """

    # Module system
    Module = _Module
    Container = _Container
    Sequential = _Sequential
    ConcatTable = _ConcatTable
    Parallel = _Parallel

    # Layers
    Linear = _Linear
    SpatialConvolution = _SpatialConvolution
    SpatialBatchNormalization = _SpatialBatchNormalization
    Dropout = _Dropout

    # Activations
    ReLU = _ReLU
    Tanh = _Tanh
    Sigmoid = _Sigmoid
    LogSoftMax = _LogSoftMax
    SoftMax = _SoftMax
    LeakyReLU = _LeakyReLU
    ELU = _ELU

    # Reshape
    View = _View
    Identity = _Identity

    # Criterion
    Criterion = _Criterion
    MSECriterion = _MSECriterion
    ClassNLLCriterion = _ClassNLLCriterion
    CrossEntropyCriterion = _CrossEntropyCriterion
    BCECriterion = _BCECriterion
    AbsCriterion = _AbsCriterion
    SmoothL1Criterion = _SmoothL1Criterion


nn = _NNNamespace()


# Register all classes in the registry
_nn_classes = [
    ("nn.Module", _Module),
    ("nn.Container", _Container),
    ("nn.Sequential", _Sequential),
    ("nn.ConcatTable", _ConcatTable),
    ("nn.Parallel", _Parallel),
    ("nn.Linear", _Linear),
    ("nn.SpatialConvolution", _SpatialConvolution),
    ("nn.SpatialBatchNormalization", _SpatialBatchNormalization),
    ("nn.Dropout", _Dropout),
    ("nn.ReLU", _ReLU),
    ("nn.Tanh", _Tanh),
    ("nn.Sigmoid", _Sigmoid),
    ("nn.LogSoftMax", _LogSoftMax),
    ("nn.SoftMax", _SoftMax),
    ("nn.LeakyReLU", _LeakyReLU),
    ("nn.ELU", _ELU),
    ("nn.View", _View),
    ("nn.Identity", _Identity),
    ("nn.Criterion", _Criterion),
    ("nn.MSECriterion", _MSECriterion),
    ("nn.ClassNLLCriterion", _ClassNLLCriterion),
    ("nn.CrossEntropyCriterion", _CrossEntropyCriterion),
    ("nn.BCECriterion", _BCECriterion),
    ("nn.AbsCriterion", _AbsCriterion),
    ("nn.SmoothL1Criterion", _SmoothL1Criterion),
]
for _name, _cls in _nn_classes:
    class_registry.register(_name, _cls)


# =============================================================================
# Functional Optimizers - Torch7 optim.*
# =============================================================================


def _optim_sgd(
    opfunc: Callable[[torch.Tensor], Tuple[float, torch.Tensor]],
    x: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Stochastic Gradient Descent matching Torch7's optim.sgd exactly.

    Args:
        opfunc: Function(x) -> (loss, gradient).
        x: Parameter tensor (modified in-place).
        config: {learningRate, momentum, weightDecay, dampening, nesterov,
                 learningRateDecay}.
        state: Optimizer state dict (created/updated across calls).

    Returns:
        (x, [loss]) after one step.
    """
    config = config or {}
    if state is None:
        state = {}
    lr = config.get("learningRate", 1e-3)
    lrd = config.get("learningRateDecay", 0)
    wd = config.get("weightDecay", 0)
    mom = config.get("momentum", 0)
    damp = config.get("dampening", mom)
    nesterov = config.get("nesterov", False)
    state.setdefault("evalCounter", 0)

    fx, dfdx = opfunc(x)

    # Weight decay
    if wd != 0:
        dfdx = dfdx.add(x, alpha=wd)

    # Momentum
    if mom != 0:
        if "dfdx" not in state:
            state["dfdx"] = dfdx.clone()
        else:
            state["dfdx"].mul_(mom).add_(dfdx, alpha=1 - damp)
        if nesterov:
            dfdx = dfdx.add(state["dfdx"], alpha=mom)
        else:
            dfdx = state["dfdx"]

    # LR decay
    clr = lr / (1 + state["evalCounter"] * lrd)

    # Update
    x.add_(dfdx, alpha=-clr)
    state["evalCounter"] += 1

    return x, [fx]


def _optim_adam(
    opfunc: Callable[[torch.Tensor], Tuple[float, torch.Tensor]],
    x: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Adam optimizer matching Torch7's optim.adam.

    Args:
        opfunc: Function(x) -> (loss, gradient).
        x: Parameter tensor (modified in-place).
        config: {learningRate, beta1, beta2, epsilon, weightDecay}.
        state: Optimizer state dict.

    Returns:
        (x, [loss]) after one step.
    """
    config = config or {}
    if state is None:
        state = {}
    lr = config.get("learningRate", 1e-3)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.999)
    eps = config.get("epsilon", 1e-8)
    wd = config.get("weightDecay", 0)

    fx, dfdx = opfunc(x)

    if wd != 0:
        dfdx = dfdx.add(x, alpha=wd)

    state.setdefault("t", 0)
    state["t"] += 1
    t = state["t"]

    if "m" not in state:
        state["m"] = torch.zeros_like(x)
        state["v"] = torch.zeros_like(x)

    state["m"].mul_(beta1).add_(dfdx, alpha=1 - beta1)
    state["v"].mul_(beta2).addcmul_(dfdx, dfdx, value=1 - beta2)

    m_hat = state["m"] / (1 - beta1 ** t)
    v_hat = state["v"] / (1 - beta2 ** t)

    x.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

    return x, [fx]


def _optim_adagrad(
    opfunc: Callable[[torch.Tensor], Tuple[float, torch.Tensor]],
    x: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """Adagrad optimizer matching Torch7's optim.adagrad."""
    config = config or {}
    if state is None:
        state = {}
    lr = config.get("learningRate", 1e-3)
    lrd = config.get("learningRateDecay", 0)
    wd = config.get("weightDecay", 0)
    eps = config.get("epsilon", 1e-10)
    state.setdefault("evalCounter", 0)

    fx, dfdx = opfunc(x)

    if wd != 0:
        dfdx = dfdx.add(x, alpha=wd)

    clr = lr / (1 + state["evalCounter"] * lrd)

    if "sum_sq" not in state:
        state["sum_sq"] = torch.zeros_like(x)

    state["sum_sq"].addcmul_(dfdx, dfdx, value=1)
    x.add_(dfdx / (state["sum_sq"].sqrt() + eps), alpha=-clr)
    state["evalCounter"] += 1

    return x, [fx]


def _optim_rmsprop(
    opfunc: Callable[[torch.Tensor], Tuple[float, torch.Tensor]],
    x: torch.Tensor,
    config: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """RMSProp optimizer matching Torch7's optim.rmsprop."""
    config = config or {}
    if state is None:
        state = {}
    lr = config.get("learningRate", 1e-2)
    alpha = config.get("alpha", 0.99)
    eps = config.get("epsilon", 1e-8)
    wd = config.get("weightDecay", 0)

    fx, dfdx = opfunc(x)

    if wd != 0:
        dfdx = dfdx.add(x, alpha=wd)

    if "v" not in state:
        state["v"] = torch.zeros_like(x)

    state["v"].mul_(alpha).addcmul_(dfdx, dfdx, value=1 - alpha)
    x.add_(dfdx / (state["v"].sqrt() + eps), alpha=-lr)

    return x, [fx]


class _OptimNamespace:
    """Namespace mirroring ``require 'optim'`` from Torch7.

    All optimizers are functional: optim.sgd(opfunc, x, config, state).
    """

    sgd = staticmethod(_optim_sgd)
    adam = staticmethod(_optim_adam)
    adagrad = staticmethod(_optim_adagrad)
    rmsprop = staticmethod(_optim_rmsprop)


optim = _OptimNamespace()


# =============================================================================
# Serialization
# =============================================================================


def save(filepath: str, obj: Any) -> None:
    """Save an object to disk using PyTorch serialization.

    Mirrors torch.save() from Torch7 but uses Python's pickle + PyTorch format.
    """
    torch.save(obj, filepath)


def load(filepath: str, map_location: Optional[str] = None) -> Any:
    """Load an object from disk.

    Mirrors torch.load() from Torch7.
    """
    return torch.load(filepath, map_location=map_location, weights_only=False)


# =============================================================================
# Utilities - Timer (tic/toc), paths
# =============================================================================


class Timer:
    """MATLAB/Torch7-style tic/toc timer.

    Mirrors sys.tic() / sys.toc() from Torch7.

    Example:
        >>> timer = Timer()
        >>> timer.tic()
        >>> # ... work ...
        >>> elapsed = timer.toc()
    """

    def __init__(self):
        self._start: Optional[float] = None

    def tic(self) -> None:
        """Start / reset the timer."""
        self._start = time.perf_counter()

    def toc(self) -> float:
        """Return seconds elapsed since last tic()."""
        if self._start is None:
            raise RuntimeError("Timer.tic() must be called before toc()")
        return time.perf_counter() - self._start

    def __repr__(self) -> str:
        if self._start is not None:
            return f"Timer(running, {self.toc():.4f}s elapsed)"
        return "Timer(stopped)"


_GLOBAL_TIMER = Timer()


def tic() -> None:
    """Start the global timer (mirrors sys.tic)."""
    _GLOBAL_TIMER.tic()


def toc() -> float:
    """Return seconds since last tic() on the global timer (mirrors sys.toc)."""
    return _GLOBAL_TIMER.toc()


# =============================================================================
# Paths utility - mirrors Torch7's paths library
# =============================================================================


class _PathsNamespace:
    """File-system utilities mirroring Torch7's paths library."""

    @staticmethod
    def home() -> str:
        """Return the user's home directory."""
        return os.path.expanduser("~")

    @staticmethod
    def concat(*parts: str) -> str:
        """Join path components (like paths.concat)."""
        return os.path.join(*parts)

    @staticmethod
    def filep(path: str) -> bool:
        """True if *path* is an existing file."""
        return os.path.isfile(path)

    @staticmethod
    def dirp(path: str) -> bool:
        """True if *path* is an existing directory."""
        return os.path.isdir(path)

    @staticmethod
    def basename(path: str) -> str:
        return os.path.basename(path)

    @staticmethod
    def dirname(path: str) -> str:
        return os.path.dirname(path)

    @staticmethod
    def extname(path: str) -> str:
        """Return the file extension including the dot."""
        return os.path.splitext(path)[1]

    @staticmethod
    def files(directory: str, pattern: str = "*") -> List[str]:
        """List files in *directory* matching a glob *pattern*."""
        import glob as _glob

        return sorted(_glob.glob(os.path.join(directory, pattern)))

    @staticmethod
    def mkdir(path: str) -> None:
        """Create directory (and parents) if it doesn't exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def rmall(path: str) -> None:
        """Remove a file or empty directory."""
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            os.rmdir(path)


paths = _PathsNamespace()
