"""Tests for torch9.rl module."""

import pytest
import torch


def test_environment():
    """Test Environment initialization."""
    from torch9.rl import Environment

    env = Environment(env_name="CartPole-v1")
    assert env.env_name == "CartPole-v1"


def test_environment_reset():
    """Test Environment reset method."""
    from torch9.rl import Environment

    env = Environment()
    state = env.reset()
    assert isinstance(state, torch.Tensor)


def test_environment_step():
    """Test Environment step method."""
    from torch9.rl import Environment

    env = Environment()
    env.reset()
    next_state, reward, done, info = env.step(0)

    assert isinstance(next_state, torch.Tensor)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_policy():
    """Test Policy initialization."""
    from torch9.rl import Policy

    policy = Policy(state_dim=4, action_dim=2)
    assert policy.state_dim == 4
    assert policy.action_dim == 2


def test_policy_call():
    """Test Policy action selection."""
    from torch9.rl import Policy

    policy = Policy(state_dim=4, action_dim=2)
    state = torch.randn(4)
    action = policy(state)
    assert isinstance(action, int)


def test_agent():
    """Test Agent initialization."""
    from torch9.rl import Agent, Environment, Policy

    policy = Policy(state_dim=4, action_dim=2)
    env = Environment()
    agent = Agent(policy, env)

    assert agent.policy == policy
    assert agent.env == env


def test_agent_train():
    """Test Agent train method."""
    from torch9.rl import Agent, Environment, Policy

    policy = Policy(state_dim=4, action_dim=2)
    env = Environment()
    agent = Agent(policy, env)

    # Should not raise exception
    agent.train(num_episodes=10)
