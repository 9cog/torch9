"""
torch9.algo - Master Algorithm Framework

A powerful framework for building, composing, and orchestrating machine learning
algorithms and pipelines. Provides building blocks for complex ML workflows.

Features:
- Algorithm: Base class for all algorithms with lifecycle management
- Pipeline: Composable processing pipelines with branching/merging
- Step: Individual processing steps with validation
- Scheduler: Algorithm scheduling and execution control
- Registry: Algorithm discovery and registration

Example:
    >>> from torch9.algo import Algorithm, Pipeline, Step, Registry
    >>>
    >>> @Registry.register("preprocessor")
    ... class Preprocessor(Algorithm):
    ...     def forward(self, x):
    ...         return normalize(x)
    >>>
    >>> pipeline = Pipeline([
    ...     Step(Preprocessor(), name="preprocess"),
    ...     Step(model, name="model"),
    ...     Step(postprocess, name="postprocess"),
    ... ])
    >>> output = pipeline(input_data)
"""

from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)
import functools
import time
import warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
from collections import OrderedDict

import torch
import torch.nn as nn

__all__ = [
    # Core
    "Algorithm",
    "Step",
    "Pipeline",
    # Execution
    "Executor",
    "ParallelExecutor",
    "SequentialExecutor",
    # Scheduling
    "Scheduler",
    "Schedule",
    "SchedulePolicy",
    # Registry
    "Registry",
    "register",
    # Flow Control
    "Branch",
    "Merge",
    "Conditional",
    "Loop",
    # State
    "AlgorithmState",
    "Context",
    # Metrics
    "Metrics",
    "Timer",
    # Decorators
    "algorithm",
    "step",
    "cached",
]

T = TypeVar("T")
InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


# =============================================================================
# Algorithm State
# =============================================================================


class AlgorithmState(Enum):
    """State of an algorithm in its lifecycle."""

    CREATED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class Context:
    """
    Execution context passed through pipeline steps.

    Contains metadata, intermediate results, and configuration.
    """

    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    errors: List[Exception] = field(default_factory=list)
    _step_outputs: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set_output(self, step_name: str, output: Any) -> None:
        """Store output from a named step."""
        self._step_outputs[step_name] = output

    def get_output(self, step_name: str) -> Any:
        """Retrieve output from a named step."""
        return self._step_outputs.get(step_name)

    def add_error(self, error: Exception) -> None:
        """Record an error."""
        self.errors.append(error)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0

    def clone(self) -> "Context":
        """Create a copy of this context."""
        return Context(
            data=self.data.copy(),
            metadata=self.metadata.copy(),
            config=self.config.copy(),
            errors=self.errors.copy(),
            _step_outputs=self._step_outputs.copy(),
        )


# =============================================================================
# Metrics & Timing
# =============================================================================


@dataclass
class Metrics:
    """Collection of algorithm metrics."""

    execution_time: float = 0.0
    step_times: Dict[str, float] = field(default_factory=dict)
    memory_used: int = 0
    iterations: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)

    def record_step(self, name: str, duration: float) -> None:
        """Record timing for a step."""
        self.step_times[name] = duration
        self.execution_time += duration

    def record_custom(self, name: str, value: Any) -> None:
        """Record custom metric."""
        self.custom[name] = value

    def summary(self) -> str:
        """Generate metrics summary."""
        lines = [
            f"Total Time: {self.execution_time:.4f}s",
            f"Iterations: {self.iterations}",
        ]
        if self.step_times:
            lines.append("Step Times:")
            for name, time in self.step_times.items():
                lines.append(f"  {name}: {time:.4f}s")
        return "\n".join(lines)


class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        self._start: Optional[float] = None
        self._end: Optional[float] = None
        self._laps: List[float] = []

    def start(self) -> "Timer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        self._end = time.perf_counter()
        return self.elapsed

    def lap(self) -> float:
        """Record a lap time."""
        lap_time = time.perf_counter() - (self._start or 0)
        self._laps.append(lap_time)
        return lap_time

    @property
    def elapsed(self) -> float:
        """Get elapsed time."""
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


# =============================================================================
# Base Algorithm
# =============================================================================


class Algorithm(nn.Module, ABC):
    """
    Base class for all algorithms in the framework.

    Provides lifecycle management, state tracking, and metrics collection.

    Example:
        >>> class MyAlgorithm(Algorithm):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.linear = nn.Linear(10, 5)
        ...
        ...     def forward(self, x):
        ...         return self.linear(x)
        ...
        ...     def validate_input(self, x):
        ...         assert x.dim() == 2, "Expected 2D input"
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self._name = name or self.__class__.__name__
        self._state = AlgorithmState.CREATED
        self._metrics = Metrics()
        self._context: Optional[Context] = None

    @property
    def name(self) -> str:
        """Algorithm name."""
        return self._name

    @property
    def state(self) -> AlgorithmState:
        """Current algorithm state."""
        return self._state

    @property
    def metrics(self) -> Metrics:
        """Algorithm metrics."""
        return self._metrics

    def initialize(self, **kwargs) -> None:
        """
        Initialize algorithm with configuration.

        Override this method to perform setup operations.
        """
        self._state = AlgorithmState.INITIALIZED

    def validate_input(self, *args, **kwargs) -> None:
        """
        Validate input before processing.

        Override to add input validation.
        Raises ValueError or TypeError for invalid inputs.
        """
        pass

    def validate_output(self, output: Any) -> None:
        """
        Validate output after processing.

        Override to add output validation.
        """
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Main algorithm logic.

        Must be implemented by subclasses.
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Execute algorithm with timing and validation."""
        timer = Timer().start()
        self._state = AlgorithmState.RUNNING

        try:
            # Validate input
            self.validate_input(*args, **kwargs)

            # Execute
            result = self.forward(*args, **kwargs)

            # Validate output
            self.validate_output(result)

            self._state = AlgorithmState.COMPLETED
            return result

        except Exception as e:
            self._state = AlgorithmState.FAILED
            raise

        finally:
            self._metrics.execution_time += timer.elapsed

    def reset(self) -> None:
        """Reset algorithm state."""
        self._state = AlgorithmState.CREATED
        self._metrics = Metrics()

    def to_step(self, name: Optional[str] = None) -> "Step":
        """Convert algorithm to a pipeline step."""
        return Step(self, name=name or self.name)


# =============================================================================
# Step
# =============================================================================


class Step:
    """
    A single processing step in a pipeline.

    Wraps a callable (Algorithm, function, or nn.Module) with metadata.

    Args:
        processor: The callable to execute
        name: Step name for identification
        enabled: Whether step is active
        cache: Whether to cache results
        on_error: Error handling ('raise', 'skip', 'default')
        default: Default value when on_error='default'
    """

    def __init__(
        self,
        processor: Callable,
        name: Optional[str] = None,
        enabled: bool = True,
        cache: bool = False,
        on_error: str = "raise",
        default: Any = None,
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
    ):
        self.processor = processor
        self.name = name or getattr(processor, "__name__", str(processor))
        self.enabled = enabled
        self.cache = cache
        self.on_error = on_error
        self.default = default
        self.input_key = input_key
        self.output_key = output_key

        self._cached_result: Optional[Any] = None
        self._executed = False

    def __call__(
        self,
        *args,
        context: Optional[Context] = None,
        **kwargs,
    ) -> Any:
        """Execute the step."""
        if not self.enabled:
            return args[0] if args else None

        # Return cached result if available
        if self.cache and self._executed:
            return self._cached_result

        # Get input from context if specified
        if self.input_key and context:
            args = (context.get_output(self.input_key),)

        try:
            # Execute processor
            if isinstance(self.processor, nn.Module):
                result = self.processor(*args, **kwargs)
            else:
                result = self.processor(*args, **kwargs)

            # Store in cache
            if self.cache:
                self._cached_result = result
                self._executed = True

            # Store in context
            if self.output_key and context:
                context.set_output(self.output_key, result)
            elif context:
                context.set_output(self.name, result)

            return result

        except Exception as e:
            if self.on_error == "raise":
                raise
            elif self.on_error == "skip":
                return args[0] if args else None
            elif self.on_error == "default":
                return self.default
            else:
                raise ValueError(f"Unknown on_error mode: {self.on_error}")

    def reset(self) -> None:
        """Reset step state."""
        self._cached_result = None
        self._executed = False

    def __repr__(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"Step({self.name}, {status})"


# =============================================================================
# Pipeline
# =============================================================================


class Pipeline(nn.Module):
    """
    Composable processing pipeline.

    Chains multiple steps together with automatic data flow.

    Example:
        >>> pipeline = Pipeline([
        ...     Step(preprocess, name="preprocess"),
        ...     Step(model, name="model"),
        ...     Step(postprocess, name="postprocess"),
        ... ])
        >>> output = pipeline(input_data)

        >>> # With branching
        >>> pipeline = Pipeline([
        ...     Step(shared_encoder, name="encoder"),
        ...     Branch([
        ...         Step(classifier, name="classify"),
        ...         Step(detector, name="detect"),
        ...     ]),
        ...     Merge(strategy="concat"),
        ... ])
    """

    def __init__(
        self,
        steps: Optional[Sequence[Union[Step, Callable]]] = None,
        name: str = "pipeline",
    ):
        super().__init__()
        self._name = name
        self._steps: nn.ModuleList = nn.ModuleList()
        self._step_list: List[Step] = []
        self._metrics = Metrics()
        self._context: Optional[Context] = None

        if steps:
            for step in steps:
                self.add_step(step)

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    def add_step(self, step: Union[Step, Callable]) -> "Pipeline":
        """Add a step to the pipeline."""
        if not isinstance(step, Step):
            step = Step(step)

        self._step_list.append(step)

        # Track nn.Module steps for state_dict
        if isinstance(step.processor, nn.Module):
            self._steps.append(step.processor)

        return self

    def remove_step(self, name: str) -> "Pipeline":
        """Remove a step by name."""
        self._step_list = [s for s in self._step_list if s.name != name]
        return self

    def get_step(self, name: str) -> Optional[Step]:
        """Get step by name."""
        for step in self._step_list:
            if step.name == name:
                return step
        return None

    def enable_step(self, name: str) -> "Pipeline":
        """Enable a step."""
        step = self.get_step(name)
        if step:
            step.enabled = True
        return self

    def disable_step(self, name: str) -> "Pipeline":
        """Disable a step."""
        step = self.get_step(name)
        if step:
            step.enabled = False
        return self

    def forward(
        self,
        x: Any,
        context: Optional[Context] = None,
    ) -> Any:
        """Execute pipeline."""
        if context is None:
            context = Context()

        self._context = context

        for step in self._step_list:
            timer = Timer().start()
            x = step(x, context=context)
            self._metrics.record_step(step.name, timer.elapsed)

        return x

    def __call__(self, x: Any, **kwargs) -> Any:
        """Execute pipeline."""
        return self.forward(x, **kwargs)

    def __len__(self) -> int:
        return len(self._step_list)

    def __iter__(self) -> Iterator[Step]:
        return iter(self._step_list)

    def __getitem__(self, idx: int) -> Step:
        return self._step_list[idx]

    def reset(self) -> None:
        """Reset pipeline and all steps."""
        self._metrics = Metrics()
        for step in self._step_list:
            step.reset()

    def summary(self) -> str:
        """Generate pipeline summary."""
        lines = [f"Pipeline: {self.name}", "=" * 40]
        for i, step in enumerate(self._step_list):
            status = "+" if step.enabled else "-"
            lines.append(f"  {status} [{i}] {step.name}")
        return "\n".join(lines)

    def compose(self, other: "Pipeline") -> "Pipeline":
        """Compose two pipelines."""
        new_pipeline = Pipeline(name=f"{self.name}+{other.name}")
        for step in self._step_list:
            new_pipeline.add_step(step)
        for step in other._step_list:
            new_pipeline.add_step(step)
        return new_pipeline

    def __add__(self, other: "Pipeline") -> "Pipeline":
        return self.compose(other)


# =============================================================================
# Flow Control
# =============================================================================


class Branch:
    """
    Execute multiple branches in parallel.

    Example:
        >>> branch = Branch([
        ...     Step(classifier, name="classify"),
        ...     Step(detector, name="detect"),
        ... ])
        >>> outputs = branch(encoded)  # Returns list of outputs
    """

    def __init__(
        self,
        branches: Sequence[Union[Step, Callable]],
        parallel: bool = False,
    ):
        self.branches = [
            b if isinstance(b, Step) else Step(b)
            for b in branches
        ]
        self.parallel = parallel

    def __call__(
        self,
        x: Any,
        context: Optional[Context] = None,
    ) -> List[Any]:
        """Execute all branches."""
        outputs = []
        for branch in self.branches:
            outputs.append(branch(x, context=context))
        return outputs


class Merge:
    """
    Merge multiple inputs into single output.

    Strategies:
        - concat: Concatenate along dimension
        - sum: Sum all inputs
        - mean: Average all inputs
        - stack: Stack along new dimension
        - dict: Return as dictionary
        - custom: Use custom merge function
    """

    def __init__(
        self,
        strategy: str = "concat",
        dim: int = -1,
        custom_fn: Optional[Callable] = None,
        names: Optional[List[str]] = None,
    ):
        self.strategy = strategy
        self.dim = dim
        self.custom_fn = custom_fn
        self.names = names

    def __call__(
        self,
        inputs: List[Any],
        context: Optional[Context] = None,
    ) -> Any:
        """Merge inputs."""
        if self.strategy == "concat":
            return torch.cat(inputs, dim=self.dim)
        elif self.strategy == "sum":
            return sum(inputs)
        elif self.strategy == "mean":
            return sum(inputs) / len(inputs)
        elif self.strategy == "stack":
            return torch.stack(inputs, dim=self.dim)
        elif self.strategy == "dict":
            names = self.names or [f"output_{i}" for i in range(len(inputs))]
            return dict(zip(names, inputs))
        elif self.strategy == "custom" and self.custom_fn:
            return self.custom_fn(inputs)
        else:
            raise ValueError(f"Unknown merge strategy: {self.strategy}")


class Conditional:
    """
    Conditionally execute steps based on a predicate.

    Example:
        >>> cond = Conditional(
        ...     predicate=lambda x: x.shape[0] > 32,
        ...     if_true=Step(batch_process),
        ...     if_false=Step(single_process),
        ... )
    """

    def __init__(
        self,
        predicate: Callable[[Any], bool],
        if_true: Union[Step, Callable],
        if_false: Optional[Union[Step, Callable]] = None,
    ):
        self.predicate = predicate
        self.if_true = if_true if isinstance(if_true, Step) else Step(if_true)
        self.if_false = (
            if_false if isinstance(if_false, Step) or if_false is None
            else Step(if_false)
        )

    def __call__(
        self,
        x: Any,
        context: Optional[Context] = None,
    ) -> Any:
        """Execute conditional."""
        if self.predicate(x):
            return self.if_true(x, context=context)
        elif self.if_false:
            return self.if_false(x, context=context)
        return x


class Loop:
    """
    Execute a step repeatedly.

    Example:
        >>> loop = Loop(
        ...     step=Step(refine),
        ...     iterations=5,
        ...     until=lambda x: x.std() < 0.1,
        ... )
    """

    def __init__(
        self,
        step: Union[Step, Callable],
        iterations: Optional[int] = None,
        until: Optional[Callable[[Any], bool]] = None,
        max_iterations: int = 1000,
    ):
        self.step = step if isinstance(step, Step) else Step(step)
        self.iterations = iterations
        self.until = until
        self.max_iterations = max_iterations

    def __call__(
        self,
        x: Any,
        context: Optional[Context] = None,
    ) -> Any:
        """Execute loop."""
        count = 0

        while True:
            # Check iteration limit
            if self.iterations is not None and count >= self.iterations:
                break
            if count >= self.max_iterations:
                warnings.warn(f"Loop reached max_iterations ({self.max_iterations})")
                break

            # Execute step
            x = self.step(x, context=context)
            count += 1

            # Check termination condition
            if self.until is not None and self.until(x):
                break

        return x


# =============================================================================
# Executors
# =============================================================================


class Executor(ABC):
    """Base class for algorithm executors."""

    @abstractmethod
    def execute(
        self,
        algorithm: Algorithm,
        *args,
        **kwargs,
    ) -> Any:
        """Execute an algorithm."""
        pass


class SequentialExecutor(Executor):
    """Execute algorithms sequentially."""

    def execute(
        self,
        algorithm: Algorithm,
        *args,
        **kwargs,
    ) -> Any:
        return algorithm(*args, **kwargs)


class ParallelExecutor(Executor):
    """Execute multiple algorithms in parallel using threads."""

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers

    def execute(
        self,
        algorithms: Sequence[Algorithm],
        inputs: Sequence[Any],
    ) -> List[Any]:
        """Execute algorithms in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(algo, inp)
                for algo, inp in zip(algorithms, inputs)
            ]
            return [f.result() for f in futures]


# =============================================================================
# Scheduling
# =============================================================================


class SchedulePolicy(Enum):
    """Scheduling policies for algorithm execution."""

    FIFO = auto()  # First in, first out
    LIFO = auto()  # Last in, first out
    PRIORITY = auto()  # Priority-based
    ROUND_ROBIN = auto()  # Round-robin


@dataclass
class Schedule:
    """Schedule entry for an algorithm."""

    algorithm: Algorithm
    priority: int = 0
    delay: float = 0.0
    repeat: int = 1
    args: Tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """
    Algorithm scheduler for managing execution order.

    Example:
        >>> scheduler = Scheduler()
        >>> scheduler.add(preprocess, priority=1)
        >>> scheduler.add(model, priority=2)
        >>> scheduler.add(postprocess, priority=3)
        >>> results = scheduler.run(input_data)
    """

    def __init__(
        self,
        policy: SchedulePolicy = SchedulePolicy.FIFO,
        executor: Optional[Executor] = None,
    ):
        self.policy = policy
        self.executor = executor or SequentialExecutor()
        self._schedules: List[Schedule] = []

    def add(
        self,
        algorithm: Algorithm,
        priority: int = 0,
        delay: float = 0.0,
        repeat: int = 1,
        **kwargs,
    ) -> "Scheduler":
        """Add algorithm to schedule."""
        self._schedules.append(Schedule(
            algorithm=algorithm,
            priority=priority,
            delay=delay,
            repeat=repeat,
            kwargs=kwargs,
        ))
        return self

    def run(self, x: Any) -> Any:
        """Execute all scheduled algorithms."""
        # Sort by policy
        if self.policy == SchedulePolicy.PRIORITY:
            schedules = sorted(self._schedules, key=lambda s: s.priority)
        elif self.policy == SchedulePolicy.LIFO:
            schedules = list(reversed(self._schedules))
        else:
            schedules = self._schedules

        # Execute
        for schedule in schedules:
            if schedule.delay > 0:
                time.sleep(schedule.delay)

            for _ in range(schedule.repeat):
                x = self.executor.execute(
                    schedule.algorithm,
                    x,
                    **schedule.kwargs,
                )

        return x

    def clear(self) -> None:
        """Clear all schedules."""
        self._schedules.clear()


# =============================================================================
# Registry
# =============================================================================


class Registry:
    """
    Global registry for algorithms.

    Example:
        >>> @Registry.register("preprocessor")
        ... class Preprocessor(Algorithm):
        ...     pass
        >>>
        >>> algo = Registry.get("preprocessor")()
    """

    _registry: Dict[str, Type[Algorithm]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register an algorithm."""
        def decorator(algo_cls: Type[Algorithm]) -> Type[Algorithm]:
            cls._registry[name] = algo_cls
            return algo_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type[Algorithm]]:
        """Get algorithm class by name."""
        return cls._registry.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered algorithm names."""
        return list(cls._registry.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> Algorithm:
        """Create algorithm instance by name."""
        algo_cls = cls.get(name)
        if algo_cls is None:
            raise KeyError(f"Algorithm not found: {name}")
        return algo_cls(**kwargs)

    @classmethod
    def clear(cls) -> None:
        """Clear registry."""
        cls._registry.clear()


# Convenience decorator
def register(name: str) -> Callable:
    """Decorator to register an algorithm in the global registry."""
    return Registry.register(name)


# =============================================================================
# Decorators
# =============================================================================


def algorithm(
    name: Optional[str] = None,
    register_name: Optional[str] = None,
) -> Callable:
    """
    Decorator to convert a function to an Algorithm.

    Example:
        >>> @algorithm(name="normalize")
        ... def normalize(x):
        ...     return (x - x.mean()) / x.std()
    """
    def decorator(func: Callable) -> Type[Algorithm]:
        algo_name = name or func.__name__

        class FuncAlgorithm(Algorithm):
            def __init__(self):
                super().__init__(name=algo_name)
                self._func = func

            def forward(self, *args, **kwargs):
                return self._func(*args, **kwargs)

        FuncAlgorithm.__name__ = algo_name
        FuncAlgorithm.__qualname__ = algo_name

        if register_name:
            Registry.register(register_name)(FuncAlgorithm)

        return FuncAlgorithm

    return decorator


def step(
    name: Optional[str] = None,
    cache: bool = False,
    on_error: str = "raise",
) -> Callable:
    """
    Decorator to convert a function to a Step.

    Example:
        >>> @step(name="preprocess", cache=True)
        ... def preprocess(x):
        ...     return normalize(x)
    """
    def decorator(func: Callable) -> Step:
        return Step(
            func,
            name=name or func.__name__,
            cache=cache,
            on_error=on_error,
        )
    return decorator


def cached(func: Callable) -> Callable:
    """
    Decorator to cache algorithm/step results.

    Uses tensor shapes as cache keys for efficient caching.
    """
    cache: Dict[Tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create cache key from tensor shapes
        key_parts = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                key_parts.append(tuple(arg.shape))
            else:
                key_parts.append(str(arg))
        key = tuple(key_parts)

        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    wrapper.cache = cache
    wrapper.clear_cache = lambda: cache.clear()
    return wrapper
