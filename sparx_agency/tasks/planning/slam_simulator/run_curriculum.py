"""SLAM environment demo with curriculum learning and discrete actions."""

from sparx_agency.tasks.planning.slam_simulator import SLAMEnv, CurriculumWrapper, DiscreteActionWrapper

# Base environment
env = SLAMEnv(
    width=32,
    height=32,
    num_agents=1,
    max_steps=1000,
    render_mode='human'
)

# Add curriculum (only 10x10 hidden area to explore)
env = CurriculumWrapper(env, hidden_size=10, random_position=True)

# Convert to discrete actions (for DQN compatibility)
env = DiscreteActionWrapper(env)

obs, info = env.reset(seed=42)
print(f"Hidden area: {info['curriculum']['hidden_size']}x{info['curriculum']['hidden_size']}")
print(f"Hidden position: {info['curriculum']['hidden_position']}")
print(f"Action space: {env.action_space}")

for step in range(300):
    action = env.action_space.sample()  # Single int action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()

    if step % 30 == 0:
        print(f"Step {step}: Progress {info['progress']*100:.1f}%, Reward: {reward:.3f}")

    if terminated or truncated:
        print(f"Done! Final progress: {info['progress']*100:.1f}%")
        break

env.close()