"""
Training Script for Custom DQN Agent with Fixed 44x44 Input
Trains on house_map_0.txt and evaluates on house_map_1.txt
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any

from planner.simulation.multi_agent_slam_gym_env import MultiAgentSLAMGymEnv
from planner.agents.dqn_slam_agent import CustomDQNAgent

# For comparison baselines
from planner.agents import RandomAgent, FrontierAgent


class DQNTrainer:
    """Trainer class for Custom DQN Agent with logging and evaluation."""

    def __init__(
        self,
        train_map_path: str,
        test_map_path: str,
        num_agents: int = 3,
        save_dir: str = "./models/custom_dqn",
        log_dir: str = "./logs/custom_dqn"
    ):
        self.train_map_path = train_map_path
        self.test_map_path = test_map_path
        self.num_agents = num_agents
        self.save_dir = save_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Training metrics
        self.train_rewards = []
        self.train_progress = []
        self.train_steps = []
        self.eval_rewards = []
        self.eval_progress = []
        self.eval_steps = []

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_env(self, map_path: str, render_mode: str = None) -> MultiAgentSLAMGymEnv:
        """Create environment with specified map."""
        # Load map to get dimensions
        map_data = np.loadtxt(map_path, dtype=np.int8)
        height, width = map_data.shape

        print(f"Loading map from: {map_path}")
        print(f"Map dimensions: {width}x{height}")

        env = MultiAgentSLAMGymEnv(
            width=width,
            height=height,
            num_drones=self.num_agents,
            num_entry_points=self.num_agents,
            camera_range=3,
            fov=30,
            max_steps=1000,
            render_mode=render_mode,
            randomize=False,  # Use fixed map
            map_path=map_path
        )

        return env

    def train(
        self,
        num_episodes: int = 10_000,
        eval_frequency: int = 20,
        save_frequency: int = 50,
        render_frequency: int = 100,
        render_eval: bool = True,
        render_training: bool = False,
        batch_size: int = 64,
        learning_rate: float = 1e-4
    ):
        """Train the Custom DQN agent."""
        print("=" * 80)
        print("CUSTOM DQN TRAINING")
        print("=" * 80)
        print(f"Training map: {self.train_map_path}")
        print(f"Test map: {self.test_map_path}")
        print(f"Number of agents: {self.num_agents}")
        print(f"Number of episodes: {num_episodes}")
        print(f"Render training: {render_training}")
        print(f"Render evaluation: {render_eval}")
        print(f"Timestamp: {self.timestamp}")
        print("=" * 80)

        # Create training environment
        train_env = self.create_env(self.train_map_path)

        # Create agent
        agent = CustomDQNAgent(
            num_agents=self.num_agents,
            learning_rate=learning_rate,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            buffer_size=20000,
            batch_size=batch_size,
            update_frequency=4,
            target_update_frequency=100
        )

        print("\nStarting training...")
        best_eval_progress = 0.0

        for episode in range(num_episodes):
            # Determine if we should render this episode
            should_render = False

            # Always render first and last episodes
            if episode == 0 or episode == num_episodes - 1:
                should_render = True
            # Render at specified frequency
            elif episode % render_frequency == 0 and episode > 0:
                should_render = True
            # Render during training if enabled
            elif render_training:
                should_render = True

            if should_render:
                train_env.render_mode = 'human'
            else:
                train_env.render_mode = None

            # Training episode
            episode_reward, episode_steps, episode_progress = self.run_episode(
                train_env, agent, training=True
            )

            self.train_rewards.append(episode_reward)
            self.train_steps.append(episode_steps)
            self.train_progress.append(episode_progress)

            # Logging
            if episode % 10 == 0:
                avg_reward = np.mean(self.train_rewards[-10:])
                avg_progress = np.mean(self.train_progress[-10:])
                avg_steps = np.mean(self.train_steps[-10:])

                print(f"\nEpisode {episode:4d} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Progress: {avg_progress:5.1%} | "
                      f"Avg Steps: {avg_steps:6.1f} | "
                      f"ε: {agent.epsilon:.3f}")

            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_reward, eval_steps, eval_progress = self.evaluate(
                    agent,
                    num_episodes=3,
                    render=render_eval
                )

                self.eval_rewards.append(eval_reward)
                self.eval_steps.append(eval_steps)
                self.eval_progress.append(eval_progress)

                print(f"  [EVAL on test map] "
                      f"Reward: {eval_reward:.2f} | "
                      f"Progress: {eval_progress:.1%} | "
                      f"Steps: {eval_steps:.1f}")

                # Save best model
                if eval_progress > best_eval_progress:
                    best_eval_progress = eval_progress
                    best_model_path = os.path.join(
                        self.save_dir,
                        f"best_model_{self.timestamp}.pt"
                    )
                    agent.save(best_model_path)
                    print(f"  [NEW BEST] Saved to {best_model_path}")

            # Regular save
            if episode % save_frequency == 0 and episode > 0:
                model_path = os.path.join(
                    self.save_dir,
                    f"checkpoint_ep{episode}_{self.timestamp}.pt"
                )
                agent.save(model_path)
                print(f"  [CHECKPOINT] Saved to {model_path}")

            # Close render window
            if train_env.render_mode == 'human':
                train_env.render_mode = None

        # Final save
        final_model_path = os.path.join(
            self.save_dir,
            f"final_model_{self.timestamp}.pt"
        )
        agent.save(final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}")

        # Plot training curves
        self.plot_training_curves()

        # Final evaluation with rendering
        print("\nFinal evaluation on test map with rendering...")
        test_env = self.create_env(self.test_map_path, render_mode='human')
        final_reward, final_steps, final_progress = self.run_episode(
            test_env, agent, training=False
        )
        print(f"Final test performance - "
              f"Reward: {final_reward:.2f} | "
              f"Progress: {final_progress:.1%} | "
              f"Steps: {final_steps}")

        # Keep window open
        # for _ in range(50):
        #     test_env.render()
        #     time.sleep(0.1)

        train_env.close()
        test_env.close()

        return agent

    def run_episode(
        self,
        env: MultiAgentSLAMGymEnv,
        agent: CustomDQNAgent,
        training: bool = True
    ) -> Tuple[float, int, float]:
        """Run a single episode."""
        observations, info = env.reset()
        if not training:
            agent.reset()  # Reset epsilon for evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0  # No exploration during evaluation

        total_reward = 0
        steps = 0

        while steps < env.max_steps:
            # Get actions
            actions = agent.get_actions(observations, info)

            # Step environment
            next_observations, rewards, dones, truncated, next_info = env.step(actions)

            # Update agent if training
            if training:
                agent.update(
                    observations, actions, rewards, next_observations,
                    dones, info, next_info
                )

            # Accumulate rewards
            total_reward += sum(rewards.values())

            # # Render if enabled
            # if env.render_mode == 'human':
            #     env.render()

            # Update state
            observations = next_observations
            info = next_info
            steps += 1

            # Check termination
            if all(dones.values()) or info['exploration_progress'] >= 0.98:
                break

        if not training:
            agent.epsilon = original_epsilon  # Restore epsilon

        return total_reward, steps, info['exploration_progress']

    def evaluate(
        self,
        agent: CustomDQNAgent,
        num_episodes: int = 5,
        render: bool = False
    ) -> Tuple[float, float, float]:
        """Evaluate agent on test map."""
        test_env = self.create_env(
            self.test_map_path,
            render_mode='human' if render else None
        )

        rewards = []
        steps_list = []
        progress_list = []

        for _ in range(num_episodes):
            reward, steps, progress = self.run_episode(
                test_env, agent, training=False
            )
            rewards.append(reward)
            steps_list.append(steps)
            progress_list.append(progress)

        test_env.close()

        return np.mean(rewards), np.mean(steps_list), np.mean(progress_list)

    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Training rewards
        ax = axes[0, 0]
        ax.plot(self.train_rewards, alpha=0.3, color='blue')
        if len(self.train_rewards) > 20:
            ax.plot(np.convolve(self.train_rewards, np.ones(20)/20, mode='valid'),
                   'b-', linewidth=2, label='20-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Training Rewards')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Training progress
        ax = axes[1, 0]
        ax.plot(np.array(self.train_progress) * 100, alpha=0.3, color='green')
        if len(self.train_progress) > 20:
            ax.plot(np.convolve(self.train_progress, np.ones(20)/20, mode='valid') * 100,
                   'g-', linewidth=2, label='20-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Progress (%)')
        ax.set_title('Training Progress')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Training steps
        ax = axes[2, 0]
        ax.plot(self.train_steps, alpha=0.3, color='orange')
        if len(self.train_steps) > 20:
            ax.plot(np.convolve(self.train_steps, np.ones(20)/20, mode='valid'),
                   'orange', linewidth=2, label='20-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Training Episode Length')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Evaluation metrics
        if self.eval_rewards:
            eval_episodes = list(range(20, len(self.train_rewards) + 1, 20))[:len(self.eval_rewards)]

            # Eval rewards
            ax = axes[0, 1]
            ax.plot(eval_episodes, self.eval_rewards, 'ro-', markersize=8, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Reward')
            ax.set_title('Evaluation Rewards (Test Map)')
            ax.grid(True, alpha=0.3)

            # Eval progress
            ax = axes[1, 1]
            ax.plot(eval_episodes, np.array(self.eval_progress) * 100, 'go-',
                   markersize=8, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Exploration Progress (%)')
            ax.set_title('Evaluation Progress (Test Map)')
            ax.grid(True, alpha=0.3)

            # Eval steps
            ax = axes[2, 1]
            ax.plot(eval_episodes, self.eval_steps, 'o-', color='orange',
                   markersize=8, linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
            ax.set_title('Evaluation Episode Length (Test Map)')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Custom DQN Training Results - {self.timestamp}', fontsize=16)
        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(
            self.log_dir,
            f'training_curves_{self.timestamp}.png'
        )
        plt.savefig(plot_path, dpi=150)
        plt.show()
        print(f"\nTraining curves saved to {plot_path}")


def compare_with_baselines(
    train_map: str,
    test_map: str,
    trained_model_path: str = None
):
    """Compare trained DQN with baseline agents."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    # Create test environment
    map_data = np.loadtxt(test_map, dtype=np.int8)
    height, width = map_data.shape

    env = MultiAgentSLAMGymEnv(
        width=width,
        height=height,
        num_drones=3,
        num_entry_points=3,
        camera_range=10,
        fov=60,
        max_steps=1000,
        render_mode=None,
        randomize=False,
        map_path=test_map
    )

    # Initialize agents
    agents = {
        'Random': RandomAgent(num_agents=3),
        'Frontier': FrontierAgent(num_agents=3, camera_range=10),
        'DQN (Untrained)': CustomDQNAgent(num_agents=3, epsilon_start=0.0),
    }

    # Load trained DQN if available
    if trained_model_path and os.path.exists(trained_model_path):
        trained_dqn = CustomDQNAgent(num_agents=3, epsilon_start=0.0)
        trained_dqn.load(trained_model_path)
        agents['DQN (Trained)'] = trained_dqn
        print(f"Loaded trained model from {trained_model_path}")

    # Results storage
    results = {name: {'rewards': [], 'steps': [], 'progress': []}
               for name in agents}

    num_episodes = 10
    print(f"\nRunning {num_episodes} episodes for each agent on test map...")

    # Test each agent
    for agent_name, agent in agents.items():
        print(f"\nTesting {agent_name}...")

        for episode in range(num_episodes):
            observations, info = env.reset()
            agent.reset()

            total_reward = 0
            steps = 0

            while steps < env.max_steps:
                actions = agent.get_actions(observations, info)
                observations, rewards, dones, truncated, info = env.step(actions)

                total_reward += sum(rewards.values())
                steps += 1

                if all(dones.values()) or info['exploration_progress'] >= 0.98:
                    break

            results[agent_name]['rewards'].append(total_reward)
            results[agent_name]['steps'].append(steps)
            results[agent_name]['progress'].append(info['exploration_progress'])

            print(f"  Episode {episode+1}: "
                  f"Reward={total_reward:.2f}, "
                  f"Progress={info['exploration_progress']:.1%}, "
                  f"Steps={steps}")

    env.close()

    # Print summary
    print("\n" + "-" * 80)
    print("SUMMARY (mean ± std)")
    print("-" * 80)

    for agent_name in agents:
        rewards = results[agent_name]['rewards']
        steps = results[agent_name]['steps']
        progress = results[agent_name]['progress']

        print(f"\n{agent_name}:")
        print(f"  Reward:   {np.mean(rewards):7.2f} ± {np.std(rewards):6.2f}")
        print(f"  Steps:    {np.mean(steps):7.1f} ± {np.std(steps):6.1f}")
        print(f"  Progress: {np.mean(progress)*100:6.1f}% ± {np.std(progress)*100:5.1f}%")

    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    agent_names = list(agents.keys())
    x_pos = np.arange(len(agent_names))

    # Rewards
    means = [np.mean(results[name]['rewards']) for name in agent_names]
    stds = [np.std(results[name]['rewards']) for name in agent_names]
    ax1.bar(x_pos, means, yerr=stds, capsize=5)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(agent_names, rotation=45, ha='right')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Reward Comparison')
    ax1.grid(True, alpha=0.3)

    # Steps
    means = [np.mean(results[name]['steps']) for name in agent_names]
    stds = [np.std(results[name]['steps']) for name in agent_names]
    ax2.bar(x_pos, means, yerr=stds, capsize=5, color='orange')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(agent_names, rotation=45, ha='right')
    ax2.set_ylabel('Average Steps')
    ax2.set_title('Episode Length Comparison')
    ax2.grid(True, alpha=0.3)

    # Progress
    means = [np.mean(results[name]['progress']) * 100 for name in agent_names]
    stds = [np.std(results[name]['progress']) * 100 for name in agent_names]
    ax3.bar(x_pos, means, yerr=stds, capsize=5, color='green')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(agent_names, rotation=45, ha='right')
    ax3.set_ylabel('Exploration Progress (%)')
    ax3.set_title('Exploration Comparison')
    ax3.grid(True, alpha=0.3)

    plt.suptitle('Agent Performance Comparison on Test Map', fontsize=16)
    plt.tight_layout()
    plt.savefig('baseline_comparison.png', dpi=150)
    plt.show()


def visualize_single_episode(
    train_map: str,
    test_map: str,
    model_path: str = None,
    use_test_map: bool = True
):
    """Visualize a single episode with a trained model."""
    print("\n" + "=" * 80)
    print("SINGLE EPISODE VISUALIZATION")
    print("=" * 80)

    # Choose map
    map_path = test_map if use_test_map else train_map
    map_name = "Test Map" if use_test_map else "Training Map"

    print(f"Visualizing on: {map_name}")
    print(f"Map path: {map_path}")

    # Load map dimensions
    map_data = np.loadtxt(map_path, dtype=np.int8)
    height, width = map_data.shape

    # Create environment with rendering
    env = MultiAgentSLAMGymEnv(
        width=width,
        height=height,
        num_drones=3,
        num_entry_points=3,
        camera_range=10,
        fov=60,
        max_steps=1000,
        render_mode='human',
        randomize=False,
        map_path=map_path
    )

    # Create agent
    agent = CustomDQNAgent(num_agents=3, epsilon_start=0.0)  # No exploration

    # Load trained model if provided
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from: {model_path}")
        agent_name = "Trained DQN"
    else:
        print("Using untrained DQN agent")
        agent_name = "Untrained DQN"

    # Run episode
    print(f"\nRunning {agent_name} on {map_name}...")
    print("Close the window to end the episode early\n")

    observations, info = env.reset()
    agent.reset()

    total_reward = 0
    steps = 0
    action_counts = {i: {'FORWARD': 0, 'TURN_LEFT': 0, 'TURN_RIGHT': 0, 'STAY': 0}
                    for i in range(3)}

    # Episode loop
    running = True
    while running and steps < env.max_steps:
        # Get actions
        actions = agent.get_actions(observations, info)

        # Log actions periodically
        if steps % 50 == 0:
            print(f"Step {steps:4d} | Progress: {info['exploration_progress']:5.1%} | ", end="")
            for agent_id, action in actions.items():
                if observations[agent_id]['active']:
                    action_name = ['FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STAY'][action]
                    print(f"D{agent_id}: {action_name} ", end="")
                    action_counts[agent_id][action_name] += 1
            print()

        # Step environment
        next_observations, rewards, dones, truncated, next_info = env.step(actions)

        # Update metrics
        total_reward += sum(rewards.values())

        # # Render
        # env.render()

        # Check for window close
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update state
        observations = next_observations
        info = next_info
        steps += 1

        # Check completion
        if all(dones.values()) or info['exploration_progress'] >= 0.98:
            print(f"\nExploration completed at {info['exploration_progress']:.1%}!")
            break

    # Final statistics
    print("\n" + "-" * 80)
    print("EPISODE SUMMARY")
    print("-" * 80)
    print(f"Agent: {agent_name}")
    print(f"Map: {map_name}")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final exploration: {info['exploration_progress']:.1%}")

    print("\nAction distribution per drone:")
    for drone_id in range(3):
        if drone_id in observations and observations[drone_id]['active']:
            print(f"\nDrone {drone_id}:")
            total_actions = sum(action_counts[drone_id].values())
            for action, count in action_counts[drone_id].items():
                percentage = (count / total_actions * 100) if total_actions > 0 else 0
                print(f"  {action:10s}: {count:4d} ({percentage:5.1f}%)")

    print("\nDiscoveries per drone:")
    for drone_id, count in info['drone_discoveries'].items():
        print(f"  Drone {drone_id}: {count} tiles")

    # Keep window open for a bit
    # print("\nKeeping window open for 5 seconds...")
    # for _ in range(50):
    #     env.render()
    #     time.sleep(0.1)

    env.close()


def main():
    """Main training script."""
    # Map paths
    train_map = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_0.txt"
    test_map = "/home/nadavc/PycharmProjects/TheAgency_workspace/resources/planner/maps/house_map_0.txt"

    # Check if we should just visualize
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "visualize":
        # Just run visualization
        model_path = sys.argv[2] if len(sys.argv) > 2 else None
        use_test = sys.argv[3] != "train" if len(sys.argv) > 3 else True
        visualize_single_episode(train_map, test_map, model_path, use_test)
        return

    # Training configuration
    config = {
        'num_episodes': 20_000,
        'eval_frequency': 20,
        'save_frequency': 50,
        'render_frequency': 100,
        'render_eval': True,      # Render evaluation episodes
        'render_training': False,  # Render all training episodes (slow!)
        'batch_size': 64,
        'learning_rate': 1e-4
    }

    # Ask user about rendering preferences
    print("\n" + "=" * 80)
    print("RENDERING OPTIONS")
    print("=" * 80)
    print("1. Minimal rendering (fastest) - only key episodes")
    print("2. Regular rendering - every 100 episodes + evaluations")
    print("3. Evaluation rendering - all evaluation episodes")
    print("4. Full rendering (slowest) - all episodes")

    choice = input("\nSelect rendering option (1-4) [default: 2]: ").strip() or "2"

    if choice == "1":
        config['render_frequency'] = 1000  # Almost never
        config['render_eval'] = False
        config['render_training'] = False
    elif choice == "2":
        config['render_frequency'] = 100
        config['render_eval'] = False
        config['render_training'] = False
    elif choice == "3":
        config['render_frequency'] = 1000
        config['render_eval'] = True
        config['render_training'] = False
    elif choice == "4":
        config['render_frequency'] = 1
        config['render_eval'] = True
        config['render_training'] = True

    # Create trainer
    trainer = DQNTrainer(
        train_map_path=train_map,
        test_map_path=test_map,
        num_agents=8,
        save_dir="./models/custom_dqn",
        log_dir="./logs/custom_dqn"
    )

    # Train agent
    trained_agent = trainer.train(**config)

    # Compare with baselines
    best_model_path = os.path.join(
        trainer.save_dir,
        f"best_model_{trainer.timestamp}.pt"
    )

    compare_with_baselines(
        train_map=train_map,
        test_map=test_map,
        trained_model_path=best_model_path
    )

    # Final visualization
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Models saved in: {trainer.save_dir}")
    print(f"Logs saved in: {trainer.log_dir}")
    print(f"Best model: {best_model_path}")

    # Ask if user wants to see final visualization
    visualize = input("\nVisualize trained agent? (y/n) [default: y]: ").strip().lower()
    if visualize != 'n':
        visualize_single_episode(train_map, test_map, best_model_path, use_test_map=True)


if __name__ == "__main__":
    main()