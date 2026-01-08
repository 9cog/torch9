"""Tests for torch9.algo module - Master Algorithm Framework."""

import pytest
import torch
import torch.nn as nn


class TestContext:
    """Tests for Context class."""

    def test_data_access(self):
        """Test data access methods."""
        from torch9.algo import Context

        ctx = Context()
        ctx["key"] = "value"
        assert ctx["key"] == "value"
        assert ctx.get("missing", "default") == "default"

    def test_step_outputs(self):
        """Test step output storage."""
        from torch9.algo import Context

        ctx = Context()
        ctx.set_output("step1", torch.randn(3))
        output = ctx.get_output("step1")
        assert output is not None
        assert output.shape == (3,)

    def test_error_tracking(self):
        """Test error tracking."""
        from torch9.algo import Context

        ctx = Context()
        assert not ctx.has_errors
        ctx.add_error(ValueError("test"))
        assert ctx.has_errors
        assert len(ctx.errors) == 1

    def test_clone(self):
        """Test context cloning."""
        from torch9.algo import Context

        ctx = Context()
        ctx["key"] = "value"
        cloned = ctx.clone()
        assert cloned["key"] == "value"
        cloned["key"] = "changed"
        assert ctx["key"] == "value"


class TestMetrics:
    """Tests for Metrics class."""

    def test_record_step(self):
        """Test step timing recording."""
        from torch9.algo import Metrics

        metrics = Metrics()
        metrics.record_step("step1", 0.5)
        metrics.record_step("step2", 0.3)

        assert metrics.step_times["step1"] == 0.5
        assert metrics.step_times["step2"] == 0.3
        assert metrics.execution_time == 0.8

    def test_record_custom(self):
        """Test custom metric recording."""
        from torch9.algo import Metrics

        metrics = Metrics()
        metrics.record_custom("accuracy", 0.95)
        assert metrics.custom["accuracy"] == 0.95

    def test_summary(self):
        """Test metrics summary."""
        from torch9.algo import Metrics

        metrics = Metrics()
        metrics.record_step("step1", 0.5)
        summary = metrics.summary()
        assert "step1" in summary
        assert "0.5" in summary


class TestTimer:
    """Tests for Timer class."""

    def test_basic_timing(self):
        """Test basic timing."""
        from torch9.algo import Timer
        import time

        timer = Timer().start()
        time.sleep(0.01)
        elapsed = timer.stop()
        assert elapsed >= 0.01

    def test_lap_timing(self):
        """Test lap timing."""
        from torch9.algo import Timer
        import time

        timer = Timer().start()
        time.sleep(0.01)
        lap1 = timer.lap()
        time.sleep(0.01)
        lap2 = timer.lap()
        assert lap2 > lap1


class TestAlgorithm:
    """Tests for Algorithm base class."""

    def test_basic_algorithm(self):
        """Test basic algorithm creation."""
        from torch9.algo import Algorithm, AlgorithmState

        class SimpleAlgo(Algorithm):
            def forward(self, x):
                return x * 2

        algo = SimpleAlgo(name="simple")
        assert algo.name == "simple"
        assert algo.state == AlgorithmState.CREATED

        result = algo(torch.tensor([1, 2, 3]))
        assert result.tolist() == [2, 4, 6]
        assert algo.state == AlgorithmState.COMPLETED

    def test_algorithm_validation(self):
        """Test input validation."""
        from torch9.algo import Algorithm

        class ValidatingAlgo(Algorithm):
            def validate_input(self, x):
                if x.dim() != 2:
                    raise ValueError("Expected 2D tensor")

            def forward(self, x):
                return x

        algo = ValidatingAlgo()
        valid = torch.randn(3, 4)
        invalid = torch.randn(3)

        result = algo(valid)
        assert result.shape == (3, 4)

        with pytest.raises(ValueError):
            algo(invalid)

    def test_algorithm_metrics(self):
        """Test algorithm metrics tracking."""
        from torch9.algo import Algorithm

        class TimedAlgo(Algorithm):
            def forward(self, x):
                return x

        algo = TimedAlgo()
        algo(torch.randn(10))
        assert algo.metrics.execution_time > 0

    def test_to_step(self):
        """Test converting algorithm to step."""
        from torch9.algo import Algorithm, Step

        class SimpleAlgo(Algorithm):
            def forward(self, x):
                return x * 2

        algo = SimpleAlgo(name="simple")
        step = algo.to_step()
        assert isinstance(step, Step)
        assert step.name == "simple"


class TestStep:
    """Tests for Step class."""

    def test_basic_step(self):
        """Test basic step execution."""
        from torch9.algo import Step

        step = Step(lambda x: x * 2, name="double")
        result = step(torch.tensor([1, 2, 3]))
        assert result.tolist() == [2, 4, 6]

    def test_disabled_step(self):
        """Test disabled step passes through."""
        from torch9.algo import Step

        step = Step(lambda x: x * 2, name="double", enabled=False)
        input_tensor = torch.tensor([1, 2, 3])
        result = step(input_tensor)
        assert result.tolist() == [1, 2, 3]

    def test_cached_step(self):
        """Test step caching."""
        from torch9.algo import Step

        call_count = [0]

        def counting_fn(x):
            call_count[0] += 1
            return x * 2

        step = Step(counting_fn, cache=True)
        input_tensor = torch.tensor([1, 2, 3])

        step(input_tensor)
        step(input_tensor)
        assert call_count[0] == 1  # Only called once

    def test_error_handling_skip(self):
        """Test error handling with skip."""
        from torch9.algo import Step

        def failing_fn(x):
            raise ValueError("Test error")

        step = Step(failing_fn, on_error="skip")
        input_tensor = torch.tensor([1, 2, 3])
        result = step(input_tensor)
        assert result.tolist() == [1, 2, 3]

    def test_error_handling_default(self):
        """Test error handling with default value."""
        from torch9.algo import Step

        def failing_fn(x):
            raise ValueError("Test error")

        step = Step(failing_fn, on_error="default", default=0)
        result = step(torch.tensor([1, 2, 3]))
        assert result == 0


class TestPipeline:
    """Tests for Pipeline class."""

    def test_basic_pipeline(self):
        """Test basic pipeline execution."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline([
            Step(lambda x: x * 2, name="double"),
            Step(lambda x: x + 1, name="add_one"),
        ])

        result = pipeline(torch.tensor([1, 2, 3]))
        assert result.tolist() == [3, 5, 7]

    def test_add_remove_step(self):
        """Test adding and removing steps."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline()
        pipeline.add_step(Step(lambda x: x * 2, name="double"))
        assert len(pipeline) == 1

        pipeline.remove_step("double")
        assert len(pipeline) == 0

    def test_enable_disable_step(self):
        """Test enabling and disabling steps."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline([
            Step(lambda x: x * 2, name="double"),
        ])

        pipeline.disable_step("double")
        step = pipeline.get_step("double")
        assert not step.enabled

        pipeline.enable_step("double")
        assert step.enabled

    def test_pipeline_context(self):
        """Test pipeline with context."""
        from torch9.algo import Pipeline, Step, Context

        pipeline = Pipeline([
            Step(lambda x: x * 2, name="double"),
        ])

        ctx = Context()
        result = pipeline(torch.tensor([1, 2, 3]), context=ctx)
        assert ctx.get_output("double") is not None

    def test_pipeline_metrics(self):
        """Test pipeline metrics."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline([
            Step(lambda x: x * 2, name="step1"),
            Step(lambda x: x + 1, name="step2"),
        ])

        pipeline(torch.tensor([1, 2, 3]))
        assert "step1" in pipeline.metrics.step_times
        assert "step2" in pipeline.metrics.step_times

    def test_pipeline_compose(self):
        """Test pipeline composition."""
        from torch9.algo import Pipeline, Step

        p1 = Pipeline([Step(lambda x: x * 2, name="double")])
        p2 = Pipeline([Step(lambda x: x + 1, name="add_one")])

        combined = p1 + p2
        result = combined(torch.tensor([1, 2, 3]))
        assert result.tolist() == [3, 5, 7]

    def test_pipeline_summary(self):
        """Test pipeline summary."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline([
            Step(lambda x: x, name="step1"),
            Step(lambda x: x, name="step2"),
        ], name="my_pipeline")

        summary = pipeline.summary()
        assert "my_pipeline" in summary
        assert "step1" in summary
        assert "step2" in summary


class TestBranch:
    """Tests for Branch class."""

    def test_basic_branch(self):
        """Test basic branching."""
        from torch9.algo import Branch, Step

        branch = Branch([
            Step(lambda x: x * 2, name="double"),
            Step(lambda x: x + 1, name="add_one"),
        ])

        input_tensor = torch.tensor([1, 2, 3])
        outputs = branch(input_tensor)

        assert len(outputs) == 2
        assert outputs[0].tolist() == [2, 4, 6]
        assert outputs[1].tolist() == [2, 3, 4]


class TestMerge:
    """Tests for Merge class."""

    def test_concat_merge(self):
        """Test concatenation merge."""
        from torch9.algo import Merge

        merge = Merge(strategy="concat", dim=0)
        inputs = [torch.randn(2, 3), torch.randn(3, 3)]
        result = merge(inputs)
        assert result.shape == (5, 3)

    def test_sum_merge(self):
        """Test sum merge."""
        from torch9.algo import Merge

        merge = Merge(strategy="sum")
        inputs = [torch.ones(3), torch.ones(3)]
        result = merge(inputs)
        assert result.tolist() == [2.0, 2.0, 2.0]

    def test_mean_merge(self):
        """Test mean merge."""
        from torch9.algo import Merge

        merge = Merge(strategy="mean")
        inputs = [torch.ones(3) * 2, torch.ones(3) * 4]
        result = merge(inputs)
        assert result.tolist() == [3.0, 3.0, 3.0]

    def test_dict_merge(self):
        """Test dictionary merge."""
        from torch9.algo import Merge

        merge = Merge(strategy="dict", names=["a", "b"])
        inputs = [torch.tensor([1]), torch.tensor([2])]
        result = merge(inputs)
        assert "a" in result
        assert "b" in result


class TestConditional:
    """Tests for Conditional class."""

    def test_if_true(self):
        """Test true branch execution."""
        from torch9.algo import Conditional, Step

        cond = Conditional(
            predicate=lambda x: x.sum() > 5,
            if_true=Step(lambda x: x * 2, name="double"),
            if_false=Step(lambda x: x + 1, name="add_one"),
        )

        big_input = torch.tensor([3, 4])  # sum = 7 > 5
        result = cond(big_input)
        assert result.tolist() == [6, 8]

    def test_if_false(self):
        """Test false branch execution."""
        from torch9.algo import Conditional, Step

        cond = Conditional(
            predicate=lambda x: x.sum() > 5,
            if_true=Step(lambda x: x * 2, name="double"),
            if_false=Step(lambda x: x + 1, name="add_one"),
        )

        small_input = torch.tensor([1, 2])  # sum = 3 < 5
        result = cond(small_input)
        assert result.tolist() == [2, 3]


class TestLoop:
    """Tests for Loop class."""

    def test_fixed_iterations(self):
        """Test fixed iteration count."""
        from torch9.algo import Loop, Step

        loop = Loop(
            step=Step(lambda x: x + 1, name="add_one"),
            iterations=3,
        )

        result = loop(torch.tensor([0]))
        assert result.tolist() == [3]

    def test_until_condition(self):
        """Test termination condition."""
        from torch9.algo import Loop, Step

        loop = Loop(
            step=Step(lambda x: x * 2, name="double"),
            until=lambda x: x.item() > 100,
        )

        result = loop(torch.tensor([1.0]))
        assert result.item() > 100


class TestRegistry:
    """Tests for Registry class."""

    def test_register_and_get(self):
        """Test registration and retrieval."""
        from torch9.algo import Registry, Algorithm

        @Registry.register("test_algo")
        class TestAlgo(Algorithm):
            def forward(self, x):
                return x

        assert "test_algo" in Registry.list()
        algo_cls = Registry.get("test_algo")
        assert algo_cls is TestAlgo

        # Cleanup
        Registry.clear()

    def test_create(self):
        """Test instance creation."""
        from torch9.algo import Registry, Algorithm

        @Registry.register("create_test")
        class CreateTest(Algorithm):
            def __init__(self, multiplier=2):
                super().__init__()
                self.multiplier = multiplier

            def forward(self, x):
                return x * self.multiplier

        algo = Registry.create("create_test", multiplier=3)
        assert algo.multiplier == 3

        # Cleanup
        Registry.clear()


class TestScheduler:
    """Tests for Scheduler class."""

    def test_basic_scheduling(self):
        """Test basic scheduling."""
        from torch9.algo import Scheduler, Algorithm, SchedulePolicy

        class Doubler(Algorithm):
            def forward(self, x):
                return x * 2

        class AddOne(Algorithm):
            def forward(self, x):
                return x + 1

        scheduler = Scheduler(policy=SchedulePolicy.FIFO)
        scheduler.add(Doubler())
        scheduler.add(AddOne())

        result = scheduler.run(torch.tensor([1, 2, 3]))
        assert result.tolist() == [3, 5, 7]

    def test_priority_scheduling(self):
        """Test priority-based scheduling."""
        from torch9.algo import Scheduler, Algorithm, SchedulePolicy

        execution_order = []

        class RecordingAlgo(Algorithm):
            def __init__(self, name):
                super().__init__(name=name)
                self._name = name

            def forward(self, x):
                execution_order.append(self._name)
                return x

        scheduler = Scheduler(policy=SchedulePolicy.PRIORITY)
        scheduler.add(RecordingAlgo("second"), priority=2)
        scheduler.add(RecordingAlgo("first"), priority=1)
        scheduler.add(RecordingAlgo("third"), priority=3)

        scheduler.run(torch.tensor([1]))
        assert execution_order == ["first", "second", "third"]


class TestDecorators:
    """Tests for decorator functions."""

    def test_algorithm_decorator(self):
        """Test @algorithm decorator."""
        from torch9.algo import algorithm

        @algorithm(name="my_normalize")
        def normalize(x):
            return (x - x.mean()) / x.std()

        algo = normalize()
        result = algo(torch.randn(100))
        assert abs(result.mean()) < 0.1
        assert abs(result.std() - 1.0) < 0.1

    def test_step_decorator(self):
        """Test @step decorator."""
        from torch9.algo import step

        @step(name="double_step", cache=True)
        def double(x):
            return x * 2

        assert double.name == "double_step"
        assert double.cache

    def test_cached_decorator(self):
        """Test @cached decorator."""
        from torch9.algo import cached

        call_count = [0]

        @cached
        def expensive_fn(x):
            call_count[0] += 1
            return x * 2

        tensor = torch.randn(3, 4)
        expensive_fn(tensor)
        expensive_fn(tensor)
        assert call_count[0] == 1


class TestNNModuleIntegration:
    """Tests for nn.Module integration."""

    def test_algorithm_as_module(self):
        """Test Algorithm works as nn.Module."""
        from torch9.algo import Algorithm

        class NeuralAlgo(Algorithm):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        algo = NeuralAlgo()

        # Can call like module
        result = algo(torch.randn(3, 10))
        assert result.shape == (3, 5)

        # Has parameters
        assert len(list(algo.parameters())) > 0

    def test_pipeline_with_modules(self):
        """Test Pipeline with nn.Module steps."""
        from torch9.algo import Pipeline, Step

        pipeline = Pipeline([
            Step(nn.Linear(10, 20), name="linear1"),
            Step(nn.ReLU(), name="relu"),
            Step(nn.Linear(20, 5), name="linear2"),
        ])

        result = pipeline(torch.randn(3, 10))
        assert result.shape == (3, 5)

        # Pipeline should track module parameters
        assert len(list(pipeline.parameters())) > 0
