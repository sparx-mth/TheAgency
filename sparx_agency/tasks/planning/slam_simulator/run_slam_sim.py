from sparx_agency.tasks.planning.slam_simulator import SLAMEnv

# Create environment
env = SLAMEnv(
    width=20,
    height=20,
    num_agents=2,
    max_steps=500,
    render_mode='human'
)

obs, info = env.reset(seed=42)

for _ in range(500):
    actions = env.action_space.sample()  # Random actions
    obs, reward, terminated, truncated, info = env.step(actions)
    env.render()

    if terminated or truncated:
        break

env.close()