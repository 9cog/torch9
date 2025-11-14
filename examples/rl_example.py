"""Example: Reinforcement Learning with torch9.rl."""

from torch9 import rl


def rl_training_example():
    """Demonstrate RL training workflow."""
    
    print("=" * 60)
    print("torch9 Reinforcement Learning Example")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating Environment")
    print("-" * 40)
    env = rl.Environment(env_name="CartPole-v1")
    print(f"Environment: {env.env_name}")
    
    # Initialize policy
    print("\n2. Initializing Policy")
    print("-" * 40)
    policy = rl.Policy(state_dim=4, action_dim=2)
    print(f"Policy: state_dim={policy.state_dim}, action_dim={policy.action_dim}")
    
    # Create agent
    print("\n3. Creating Agent")
    print("-" * 40)
    agent = rl.Agent(policy=policy, env=env)
    print("Agent created successfully")
    
    # Run episode
    print("\n4. Running Sample Episode")
    print("-" * 40)
    state = env.reset()
    print(f"Initial state: {state}")
    
    total_reward = 0
    for step in range(10):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        print(f"Step {step + 1}: action={action}, reward={reward:.2f}")
        state = next_state
        if done:
            break
    
    print(f"\nTotal reward: {total_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("RL example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    rl_training_example()
