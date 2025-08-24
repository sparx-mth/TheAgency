"""
compare_agents.py - UPDATED FOR DQN

Comprehensive benchmark to compare Random, Frontier, and DQN agents on house_map_10.
Uses the exact same environment configuration as DQN training for fair comparison.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Import environment and agents
from environments.base.slam_env import MultiAgentSLAMEnv
from sensors.camera_sensor import CameraSensor
# from agents.random_agent import RandomAgent
from agents.frontier_agent import FrontierAgent

# Import for DQN
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from environments.wrappers.multidiscrete_wrapper import MultiDiscreteToDiscreteWrapper

# Import the custom feature extractor (needed for loading the model)
# Try multiple possible import paths
SLAMCNNExtractor = None
try:
    from cnn_feature_extractor import SLAMCNNExtractor
    print("  Imported SLAMCNNExtractor from cnn_feature_extractor")
except ImportError:
    try:
        import sys
        import os
        # Add the src/rl directory to path
        rl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'rl')
        if rl_path not in sys.path:
            sys.path.append(rl_path)
        from cnn_feature_extractor import SLAMCNNExtractor
        print("  Imported SLAMCNNExtractor from rl directory")
    except ImportError:
        print("  Warning: Could not import cnn_feature_extractor from any location.")
        print("  Creating a local copy for model loading...")

        # Create local copy inline
        import torch
        import torch.nn as nn
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        from gymnasium import spaces

        class SLAMCNNExtractor(BaseFeaturesExtractor):
            def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
                cnn_output_dim = 64
                other_features_dim = observation_space['positions'].shape[0] * 2 + \
                                     observation_space['facings'].shape[0] + \
                                     observation_space['active'].shape[0]

                super().__init__(observation_space, features_dim)

                self.cnn = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, cnn_output_dim),
                    nn.ReLU()
                )

                combined_dim = cnn_output_dim + other_features_dim
                self.mlp = nn.Sequential(
                    nn.Linear(combined_dim, features_dim),
                    nn.ReLU(),
                    nn.Linear(features_dim, features_dim),
                    nn.ReLU()
                )

            def forward(self, observations):
                map_data = observations['global_map'].float()
                map_data = map_data.unsqueeze(1)
                cnn_features = self.cnn(map_data)

                positions = observations['positions'].float().flatten(start_dim=1)
                facings = observations['facings'].float()
                active = observations['active'].float()

                other_features = torch.cat([positions, facings, active], dim=1)
                combined = torch.cat([cnn_features, other_features], dim=1)

                return self.mlp(combined)

        print("  Created local SLAMCNNExtractor class")


class DQNAgentWrapper:
    """Wrapper to make DQN agent compatible with our benchmark interface."""

    def __init__(self, model_path: str):
        """
        Initialize DQN agent wrapper.

        Args:
            model_path: Path to trained DQN model
        """
        self.model = DQN.load(model_path)
        self.num_agents = 1

    def setup_env(self, env):
        """Setup wrapped environment for DQN."""
        # Don't need to setup here, we'll handle wrapping in the trial
        pass

    def get_actions(self, observations, info):
        """Get actions from DQN model using wrapped environment."""
        # DQN expects single discrete action, predict it
        action, _ = self.model.predict(observations, deterministic=True)

        # Convert discrete action back to multi-agent format
        # This will be handled by the wrapper in the trial logic
        return action

    def reset(self):
        """Reset the agent."""
        pass  # DQN doesn't need state reset


def create_environment_exact_dqn_config(render=False):
    """
    Create environment with EXACT same configuration as DQN training.

    Args:
        render: Whether to render the environment

    Returns:
        Environment instance
    """
    # EXACT same sensor configuration as DQN training
    sensor = CameraSensor(
        max_range=5,  # Same as training
        fov_deg=90,  # Same as training
        num_rays=20  # Same as training
    )

    # EXACT same environment parameters as DQN training
    env = MultiAgentSLAMEnv(
        width=10,
        height=10,
        num_agents=1,
        max_steps=500,  # Same as training
        map_path="/home/user/nadav/TheAgency/resources/planner/maps/house_map_10.txt",
        randomize=False,  # Always use the same map
        render_mode='human' if render else None,
        sensor_config={0: sensor},
        # EXACT same reward structure as training
        discovery_reward=1.0,  # Same as training
        collision_penalty=-0.1,  # Same as training
        step_penalty=0.0,  # Same as training
        completion_bonus=50.0,  # Same as training
    )

    return env


def run_single_trial(agent, agent_type, trial_num, render=False, render_delay=0.0):
    """
    Run a single trial for an agent.

    Args:
        agent: The agent to test
        agent_type: Type of agent ('random', 'frontier', 'dqn')
        trial_num: Trial number
        render: Whether to render
        render_delay: Delay between frames when rendering

    Returns:
        Dictionary with trial results
    """
    # Create environment
    env = create_environment_exact_dqn_config(render=render)

    # Special handling for DQN agent - wrap the environment
    if agent_type == 'dqn':
        # Wrap environment for DQN (same as training)
        monitored_env = Monitor(env)
        wrapped_env = MultiDiscreteToDiscreteWrapper(monitored_env)

        obs, info = wrapped_env.reset()
        agent.reset()
        actual_env = env  # Keep reference to original for metrics
    else:
        obs, info = env.reset()
        agent.reset()
        actual_env = env

    # Initialize metrics
    done = False
    truncated = False
    step_count = 0
    total_reward = 0.0
    start_time = time.time()

    # Track detailed metrics
    collision_count = 0
    discovered_cells = []
    progress_history = []
    reward_history = []

    while not done and not truncated:
        # Get action based on agent type
        if agent_type == 'dqn':
            # Get discrete action from DQN
            discrete_action = agent.get_actions(obs, info)
            # Step the wrapped environment
            obs, reward, done, truncated, info = wrapped_env.step(discrete_action)
        else:
            actions = agent.get_actions(obs, info)
            obs, reward, done, truncated, info = env.step(actions)

        # Update metrics
        step_count += 1
        total_reward += reward
        reward_history.append(reward)

        # Track progress
        current_progress = info.get('progress', 0)
        progress_history.append(current_progress)

        # Track discoveries
        current_discovered = info.get('discovered_cells', 0)
        discovered_cells.append(current_discovered)

        # Track collisions
        if 'collision_counts' in info:
            collision_count = sum(info['collision_counts'])

        # Render if requested
        if render:
            actual_env.render()
            if render_delay > 0:
                time.sleep(render_delay)

    # Calculate final metrics
    completion_time = time.time() - start_time
    final_progress = info.get('progress', 0)
    final_discovered = info.get('discovered_cells', 0)
    completed = final_progress >= 0.99

    # Calculate efficiency metrics
    if step_count > 0:
        avg_reward_per_step = total_reward / step_count
        discovery_rate = final_discovered / step_count if step_count > 0 else 0
    else:
        avg_reward_per_step = 0
        discovery_rate = 0

    # Close environment
    if agent_type == 'dqn':
        wrapped_env.close()
    else:
        env.close()

    return {
        'agent_type': agent_type,
        'trial': trial_num,
        'completed': completed,
        'time': completion_time,
        'steps': step_count,
        'total_reward': total_reward,
        'avg_reward_per_step': avg_reward_per_step,
        'final_progress': final_progress,
        'final_discovered': final_discovered,
        'collisions': collision_count,
        'discovery_rate': discovery_rate,
        'progress_history': progress_history,
        'reward_history': reward_history,
    }


def run_benchmark(num_trials=10, render_first=False, render_all=False, save_results=True):
    """
    Run complete benchmark comparing all three agents.

    Args:
        num_trials: Number of trials per agent
        render_first: Whether to render the first trial of each agent
        render_all: Whether to render all trials
        save_results: Whether to save results to CSV

    Returns:
        DataFrame with all results
    """
    print("=" * 70)
    print("MULTI-AGENT COMPARISON BENCHMARK")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Map: house_map_10.txt")
    print(f"  Trials per agent: {num_trials}")
    print(f"  Camera: range=5, FOV=90°, rays=20")
    print(f"  Max steps: 500")
    print("=" * 70)

    # Initialize agents
    agents = []

    # # 1. Random Agent
    # print("\nInitializing Random Agent...")
    # random_agent = RandomAgent(num_agents=1, forward_bias=0.7, seed=42)
    # agents.append(('random', random_agent))

    # 2. Frontier Agent
    print("Initializing Frontier Agent...")
    frontier_agent = FrontierAgent(num_agents=1, camera_range=5)
    agents.append(('frontier', frontier_agent))

    # 3. DQN Agent - Updated path
    dqn_model_path = "/home/user/nadav/TheAgency/src/rl/models/dqn/interrupted_model"

    if os.path.exists(f"{dqn_model_path}.zip"):
        print("Initializing DQN Agent...")
        print(f"Found DQN model at: {dqn_model_path}.zip")
        dqn_agent = DQNAgentWrapper(dqn_model_path)
        agents.append(('dqn', dqn_agent))
    else:
        print("WARNING: DQN model not found!")
        print(f"         Searched: {dqn_model_path}.zip")
        print("         Skipping DQN agent.")

    # Run trials
    all_results = []
    total_trials = len(agents) * num_trials

    with tqdm(total=total_trials, desc="Running trials") as pbar:
        for agent_name, agent in agents:
            print(f"\n\nTesting {agent_name.upper()} Agent")
            print("-" * 40)

            for trial in range(1, num_trials + 1):
                # Render based on settings
                render = (render_first and trial == 1) or render_all

                # Run trial
                try:
                    result = run_single_trial(
                        agent,
                        agent_name,
                        trial,
                        render=render,
                        render_delay=0.05 if render else 0
                    )
                    all_results.append(result)

                    # Print progress
                    if trial == 1 or trial % 5 == 0:
                        print(f"  Trial {trial:2d}: Progress={result['final_progress'] * 100:.1f}%, "
                              f"Steps={result['steps']:3d}, Reward={result['total_reward']:.1f}")

                except Exception as e:
                    print(f"  ERROR in trial {trial}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # Add failed trial
                    all_results.append({
                        'agent_type': agent_name,
                        'trial': trial,
                        'completed': False,
                        'time': None,
                        'steps': 0,
                        'total_reward': 0,
                        'avg_reward_per_step': 0,
                        'final_progress': 0,
                        'final_discovered': 0,
                        'collisions': 0,
                        'discovery_rate': 0,
                        'progress_history': [],
                        'reward_history': [],
                    })

                pbar.update(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results if requested
    if save_results:
        os.makedirs("results", exist_ok=True)
        csv_path = "results/agent_comparison_results.csv"

        # Save main metrics
        df_save = df.drop(columns=['progress_history', 'reward_history'])
        df_save.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    return df


def create_comprehensive_visualization(df, save_path="results/comparison_plots.png"):
    """
    Create comprehensive visualization comparing all agents.

    Args:
        df: DataFrame with results
        save_path: Path to save the figure
    """
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Color palette for agents (updated for DQN)
    colors = {'random': '#FF6B6B', 'frontier': '#4ECDC4', 'dqn': '#45B7D1'}

    # 1. Success Rate Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    success_rates = df.groupby('agent_type')['completed'].mean()
    bars = ax1.bar(success_rates.index, success_rates.values,
                   color=[colors[x] for x in success_rates.index])
    ax1.set_title('Success Rate', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Success Rate')
    ax1.set_ylim([0, 1.1])
    # Add value labels on bars
    for bar, val in zip(bars, success_rates.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    # 2. Steps to Complete (violin plot)
    ax2 = fig.add_subplot(gs[0, 1])
    sns.violinplot(data=df, x='agent_type', y='steps', ax=ax2,
                   palette=colors, inner='box')
    ax2.set_title('Steps Distribution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Steps')
    ax2.set_xlabel('')

    # 3. Total Reward Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sns.boxplot(data=df, x='agent_type', y='total_reward', ax=ax3,
                palette=colors)
    ax3.set_title('Total Reward Distribution', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Total Reward')
    ax3.set_xlabel('')

    # 4. Final Progress
    ax4 = fig.add_subplot(gs[0, 3])
    df['progress_percent'] = df['final_progress'] * 100
    sns.violinplot(data=df, x='agent_type', y='progress_percent', ax=ax4,
                   palette=colors, inner='box')
    ax4.set_title('Final Progress', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Progress (%)')
    ax4.set_xlabel('')
    ax4.set_ylim([0, 105])

    # 5. Completion Time (only successful runs)
    ax5 = fig.add_subplot(gs[1, 0])
    df_completed = df[df['completed'] == True]
    if not df_completed.empty:
        sns.boxplot(data=df_completed, x='agent_type', y='time', ax=ax5,
                    palette=colors)
        ax5.set_title('Completion Time (Successful Only)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Time (seconds)')
        ax5.set_xlabel('')

    # 6. Collision Analysis
    ax6 = fig.add_subplot(gs[1, 1])
    sns.boxplot(data=df, x='agent_type', y='collisions', ax=ax6,
                palette=colors)
    ax6.set_title('Collision Count', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Collisions')
    ax6.set_xlabel('')

    # 7. Efficiency: Discovery Rate
    ax7 = fig.add_subplot(gs[1, 2])
    sns.violinplot(data=df, x='agent_type', y='discovery_rate', ax=ax7,
                   palette=colors, inner='box')
    ax7.set_title('Discovery Efficiency', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Cells Discovered per Step')
    ax7.set_xlabel('')

    # 8. Reward per Step
    ax8 = fig.add_subplot(gs[1, 3])
    sns.boxplot(data=df, x='agent_type', y='avg_reward_per_step', ax=ax8,
                palette=colors)
    ax8.set_title('Average Reward per Step', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Reward per Step')
    ax8.set_xlabel('')

    # 9. Trial-by-Trial Performance
    ax9 = fig.add_subplot(gs[2, :2])
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        ax9.plot(agent_df['trial'], agent_df['final_progress'] * 100,
                 marker='o', label=agent_type.upper(), color=colors[agent_type],
                 linewidth=2, markersize=8)
    ax9.set_title('Performance Across Trials', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Trial Number')
    ax9.set_ylabel('Final Progress (%)')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([0, 105])

    # 10. Summary Statistics Table
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('tight')
    ax10.axis('off')

    # Calculate summary statistics
    summary_data = []
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        completed_df = agent_df[agent_df['completed'] == True]

        summary_data.append([
            agent_type.upper(),
            f"{agent_df['completed'].mean():.1%}",
            f"{agent_df['final_progress'].mean():.1%}",
            f"{agent_df['steps'].mean():.0f} ± {agent_df['steps'].std():.0f}",
            f"{agent_df['total_reward'].mean():.1f} ± {agent_df['total_reward'].std():.1f}",
            f"{completed_df['time'].mean():.1f}s" if not completed_df.empty else "N/A",
            f"{agent_df['collisions'].mean():.1f}",
            f"{agent_df['discovery_rate'].mean():.3f}"
        ])

    # Create table
    col_labels = ['Agent', 'Success', 'Avg Progress', 'Steps', 'Reward', 'Time', 'Collisions', 'Discovery/Step']
    table = ax10.table(cellText=summary_data, colLabels=col_labels,
                       cellLoc='center', loc='center',
                       colColours=['#f0f0f0'] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Color code the agent column
    for i, agent_type in enumerate(df['agent_type'].unique()):
        table[(i + 1, 0)].set_facecolor(colors[agent_type])
        table[(i + 1, 0)].set_text_props(weight='bold', color='white')

    ax10.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('Multi-Agent SLAM Benchmark Comparison\nHouse Map 10 | Camera: range=5, FOV=90°',
                 fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

    return fig


def create_learning_curves(df, save_path="results/learning_curves.png"):
    """
    Create learning curves showing progress over time for each agent.

    Args:
        df: DataFrame with results
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'random': '#FF6B6B', 'frontier': '#4ECDC4', 'dqn': '#45B7D1'}

    for idx, agent_type in enumerate(df['agent_type'].unique()):
        ax = axes[idx]
        agent_df = df[df['agent_type'] == agent_type]

        # Plot progress over steps for each trial
        for trial in agent_df['trial'].unique():
            trial_data = agent_df[agent_df['trial'] == trial].iloc[0]
            if len(trial_data['progress_history']) > 0:
                progress = np.array(trial_data['progress_history']) * 100
                steps = range(len(progress))
                ax.plot(steps, progress, alpha=0.3, color=colors[agent_type])

        # Plot average progress
        all_progress = []
        max_len = max(len(row['progress_history']) for _, row in agent_df.iterrows())

        for i in range(max_len):
            step_progress = []
            for _, row in agent_df.iterrows():
                if i < len(row['progress_history']):
                    step_progress.append(row['progress_history'][i] * 100)
            if step_progress:
                all_progress.append(np.mean(step_progress))

        if all_progress:
            ax.plot(range(len(all_progress)), all_progress,
                    color=colors[agent_type], linewidth=3, label='Average')

        ax.set_title(f'{agent_type.upper()} Agent', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Progress (%)')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('Learning Curves - Progress Over Time', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Learning curves saved to {save_path}")

    return fig


def print_detailed_comparison(df):
    """
    Print detailed comparison statistics to console.

    Args:
        df: DataFrame with results
    """
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 70)

    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        completed_df = agent_df[agent_df['completed'] == True]

        print(f"\n{agent_type.upper()} AGENT:")
        print("-" * 40)
        print(f"  Success Rate: {agent_df['completed'].mean():.1%} ({completed_df.shape[0]}/{agent_df.shape[0]})")
        print(f"  Average Progress: {agent_df['final_progress'].mean():.1%} ± {agent_df['final_progress'].std():.1%}")
        print(f"  Average Steps: {agent_df['steps'].mean():.0f} ± {agent_df['steps'].std():.0f}")
        print(f"  Average Reward: {agent_df['total_reward'].mean():.1f} ± {agent_df['total_reward'].std():.1f}")

        if not completed_df.empty:
            print(f"  Completion Time: {completed_df['time'].mean():.2f}s ± {completed_df['time'].std():.2f}s")

        print(f"  Average Collisions: {agent_df['collisions'].mean():.1f} ± {agent_df['collisions'].std():.1f}")
        print(f"  Discovery Rate: {agent_df['discovery_rate'].mean():.3f} cells/step")
        print(f"  Reward Efficiency: {agent_df['avg_reward_per_step'].mean():.3f} reward/step")

        # Best and worst trials
        best_trial = agent_df.loc[agent_df['total_reward'].idxmax()]
        worst_trial = agent_df.loc[agent_df['total_reward'].idxmin()]

        print(f"\n  Best Trial:")
        print(f"    Trial #{best_trial['trial']}: Reward={best_trial['total_reward']:.1f}, "
              f"Progress={best_trial['final_progress'] * 100:.1f}%, Steps={best_trial['steps']}")

        print(f"  Worst Trial:")
        print(f"    Trial #{worst_trial['trial']}: Reward={worst_trial['total_reward']:.1f}, "
              f"Progress={worst_trial['final_progress'] * 100:.1f}%, Steps={worst_trial['steps']}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISON")
    print("=" * 70)

    # Find best agent for each metric
    metrics = {
        'Success Rate': df.groupby('agent_type')['completed'].mean(),
        'Avg Progress': df.groupby('agent_type')['final_progress'].mean(),
        'Avg Reward': df.groupby('agent_type')['total_reward'].mean(),
        'Fewest Steps': -df.groupby('agent_type')['steps'].mean(),  # Negative for min
        'Discovery Rate': df.groupby('agent_type')['discovery_rate'].mean(),
    }

    for metric_name, values in metrics.items():
        if metric_name == 'Fewest Steps':
            best_agent = values.idxmax()
            best_value = -values[best_agent]
            print(f"  {metric_name}: {best_agent.upper()} ({best_value:.0f} steps)")
        else:
            best_agent = values.idxmax()
            best_value = values[best_agent]
            if 'Rate' in metric_name:
                print(f"  {metric_name}: {best_agent.upper()} ({best_value:.3f})")
            elif 'Progress' in metric_name or 'Success' in metric_name:
                print(f"  {metric_name}: {best_agent.upper()} ({best_value:.1%})")
            else:
                print(f"  {metric_name}: {best_agent.upper()} ({best_value:.1f})")

    print("\n" + "=" * 70)


def main():
    """Main function to run the complete benchmark."""
    print("\n" + "=" * 70)
    print("SLAM AGENT COMPARISON BENCHMARK")
    print("Comparing: Random vs Frontier vs DQN")
    print("=" * 70)

    # Configuration
    num_trials = 10
    render_first = False  # Set to True to see first trial of each agent

    # Ask user for rendering preference
    print("\nRendering options:")
    print("1. No rendering (fastest)")
    print("2. Render first trial of each agent")
    print("3. Render all trials (very slow)")

    choice = input("Select option (1-3) [default=1]: ").strip() or "1"

    if choice == "2":
        render_first = True
        render_all = False
    elif choice == "3":
        render_first = False
        render_all = True
        print("\nWARNING: Rendering all trials will be very slow!")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return
    else:
        render_first = False
        render_all = False

    # Run benchmark
    print("\nStarting benchmark...")
    df = run_benchmark(num_trials=num_trials, render_first=render_first, render_all=render_all)

    # Create visualizations
    print("\nGenerating visualizations...")
    create_comprehensive_visualization(df)
    create_learning_curves(df)

    # Print detailed comparison
    print_detailed_comparison(df)

    # Show plots
    print("\nShowing plots...")
    plt.show()

    print("\nBenchmark complete!")
    print("Results saved to 'results/' directory")


if __name__ == "__main__":
    main()