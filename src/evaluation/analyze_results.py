"""
analyze_results.py

This script analyzes SLAM benchmark results and generates comprehensive visualizations.
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SLAM Benchmark Results')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory containing logs and results')
    parser.add_argument('--csv_name', type=str, default='slam_benchmark_results.csv',
                        help='Name of CSV results file')
    parser.add_argument('--output_name', type=str, default='benchmark_analysis.png',
                        help='Name of output visualization file')
    parser.add_argument('--agent_types', nargs='+', type=str, default=None,
                        help='Agent types to analyze (default: all)')
    parser.add_argument('--show_plot', action='store_true',
                        help='Show plot in addition to saving')
    return parser.parse_args()


def load_and_clean_data(csv_path):
    """
    Load and clean the benchmark data.

    Args:
        csv_path: Path to CSV file

    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df = pd.read_csv(csv_path)

    # Convert completed column to boolean if it's not already
    if 'completed' in df.columns:
        df['completed'] = df['completed'].astype(bool)

    # Calculate success rate per configuration
    success_rates = df.groupby(['map', 'drones', 'agent_type'])['completed'].mean().reset_index()
    success_rates.columns = ['map', 'drones', 'agent_type', 'success_rate']

    return df, success_rates


def create_comprehensive_analysis(df, success_rates, output_path, show_plot=False):
    """
    Create a comprehensive visualization of benchmark results.

    Args:
        df: Main dataframe with all results
        success_rates: Dataframe with success rates
        output_path: Path to save the figure
        show_plot: Whether to display the plot
    """
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 100

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))

    # Create a grid of subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Success Rate by Number of Drones
    ax1 = fig.add_subplot(gs[0, 0])
    success_by_drones = df.groupby(['drones', 'agent_type'])['completed'].mean()
    success_by_drones.unstack().plot(kind='bar', ax=ax1)
    ax1.set_title('Success Rate by Number of Drones')
    ax1.set_xlabel('Number of Drones')
    ax1.set_ylabel('Success Rate')
    ax1.legend(title='Agent Type')
    ax1.set_ylim([0, 1.1])

    # 2. Average Completion Time by Drones (only successful runs)
    ax2 = fig.add_subplot(gs[0, 1])
    completed_df = df[df['completed'] == True]
    if not completed_df.empty:
        sns.boxplot(data=completed_df, x='drones', y='time', hue='agent_type', ax=ax2)
        ax2.set_title('Completion Time Distribution (Successful Runs)')
        ax2.set_xlabel('Number of Drones')
        ax2.set_ylabel('Time (seconds)')

    # 3. Progress Distribution for Failed Runs
    ax3 = fig.add_subplot(gs[0, 2])
    failed_df = df[df['completed'] == False]
    if not failed_df.empty:
        sns.boxplot(data=failed_df, x='drones', y='progress', hue='agent_type', ax=ax3)
        ax3.set_title('Progress Distribution (Failed Runs)')
        ax3.set_xlabel('Number of Drones')
        ax3.set_ylabel('Progress (%)')
        ax3.set_ylim([0, 1])

    # 4. Average Reward by Configuration
    ax4 = fig.add_subplot(gs[1, 0])
    reward_by_config = df.groupby(['drones', 'agent_type'])['avg_reward'].mean().unstack()
    reward_by_config.plot(kind='bar', ax=ax4)
    ax4.set_title('Average Reward per Agent')
    ax4.set_xlabel('Number of Drones')
    ax4.set_ylabel('Average Reward')
    ax4.legend(title='Agent Type')

    # 5. Collision Rate Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    df['collision_rate'] = df['collisions'] / df['steps']
    collision_rates = df.groupby(['drones', 'agent_type'])['collision_rate'].mean().unstack()
    collision_rates.plot(kind='bar', ax=ax5)
    ax5.set_title('Collision Rate (Collisions per Step)')
    ax5.set_xlabel('Number of Drones')
    ax5.set_ylabel('Collision Rate')
    ax5.legend(title='Agent Type')

    # 6. Efficiency: Progress per Step
    ax6 = fig.add_subplot(gs[1, 2])
    df['efficiency'] = df['progress'] / df['steps']
    efficiency = df.groupby(['drones', 'agent_type'])['efficiency'].mean().unstack()
    efficiency.plot(kind='bar', ax=ax6)
    ax6.set_title('Exploration Efficiency (Progress per Step)')
    ax6.set_xlabel('Number of Drones')
    ax6.set_ylabel('Efficiency')
    ax6.legend(title='Agent Type')

    # 7. Performance by Map (Heatmap)
    ax7 = fig.add_subplot(gs[2, :2])
    if 'map' in df.columns:
        pivot_table = df.pivot_table(
            values='completed',
            index='map',
            columns=['drones', 'agent_type'],
            aggfunc='mean'
        )
        sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax7, vmin=0, vmax=1)
        ax7.set_title('Success Rate Heatmap by Map and Configuration')
        ax7.set_xlabel('Configuration (Drones, Agent Type)')
        ax7.set_ylabel('Map Index')

    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('tight')
    ax8.axis('off')

    # Calculate summary statistics
    summary_data = []
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        completed_agent_df = agent_df[agent_df['completed'] == True]

        summary_data.append([
            agent_type,
            f"{agent_df['completed'].mean():.1%}",
            f"{completed_agent_df['time'].mean():.1f}s" if not completed_agent_df.empty else "N/A",
            f"{agent_df['progress'].mean():.1%}",
            f"{agent_df['collisions'].mean():.1f}",
            f"{agent_df['avg_reward'].mean():.1f}"
        ])

    table = ax8.table(
        cellText=summary_data,
        colLabels=['Agent', 'Success', 'Avg Time', 'Avg Progress', 'Avg Collisions', 'Avg Reward'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax8.set_title('Summary Statistics', fontsize=12, pad=20)

    # Overall title
    fig.suptitle('SLAM Benchmark Analysis', fontsize=16, y=0.98)

    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_comparison_plot(df, output_path, show_plot=False):
    """
    Create a focused comparison plot for different agent types.

    Args:
        df: Dataframe with results
        output_path: Path to save the figure
        show_plot: Whether to display the plot
    """
    # Filter for only completed runs
    completed_df = df[df['completed'] == True].copy()

    if completed_df.empty:
        print("No completed runs found for comparison plot")
        return

    # Set style
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Time comparison
    sns.violinplot(data=completed_df, x='agent_type', y='time', hue='drones', ax=axes[0, 0])
    axes[0, 0].set_title('Completion Time by Agent Type and Drone Count')
    axes[0, 0].set_xlabel('Agent Type')
    axes[0, 0].set_ylabel('Time (seconds)')

    # 2. Speedup factor
    speedup_data = []
    for agent_type in completed_df['agent_type'].unique():
        agent_data = completed_df[completed_df['agent_type'] == agent_type]
        base_time = agent_data[agent_data['drones'] == 1]['time'].mean()

        for num_drones in sorted(agent_data['drones'].unique()):
            if num_drones > 1:
                drone_time = agent_data[agent_data['drones'] == num_drones]['time'].mean()
                speedup = base_time / drone_time if drone_time > 0 else 0
                speedup_data.append({
                    'agent_type': agent_type,
                    'drones': num_drones,
                    'speedup': speedup
                })

    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        sns.barplot(data=speedup_df, x='drones', y='speedup', hue='agent_type', ax=axes[0, 1])
        axes[0, 1].set_title('Speedup Factor (vs Single Drone)')
        axes[0, 1].set_xlabel('Number of Drones')
        axes[0, 1].set_ylabel('Speedup Factor')
        axes[0, 1].axhline(y=1, color='black', linestyle='--', alpha=0.5)

    # 3. Reward efficiency
    df['reward_per_step'] = df['total_reward'] / df['steps']
    sns.boxplot(data=df, x='agent_type', y='reward_per_step', hue='drones', ax=axes[1, 0])
    axes[1, 0].set_title('Reward Efficiency (Reward per Step)')
    axes[1, 0].set_xlabel('Agent Type')
    axes[1, 0].set_ylabel('Reward per Step')

    # 4. Success rate over iterations (learning curve)
    iteration_success = df.groupby(['iteration', 'agent_type', 'drones'])['completed'].mean().reset_index()
    for agent_type in iteration_success['agent_type'].unique():
        agent_iter = iteration_success[iteration_success['agent_type'] == agent_type]
        for num_drones in sorted(agent_iter['drones'].unique()):
            drone_iter = agent_iter[agent_iter['drones'] == num_drones]
            label = f"{agent_type} ({num_drones} drones)"
            axes[1, 1].plot(drone_iter['iteration'], drone_iter['completed'],
                            marker='o', label=label, alpha=0.7)

    axes[1, 1].set_title('Success Rate Over Iterations')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Success Rate')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].set_ylim([0, 1.1])

    plt.suptitle('Agent Performance Comparison', fontsize=14)
    plt.tight_layout()

    # Save with different name
    comparison_path = output_path.parent / f"comparison_{output_path.name}"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {comparison_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def print_summary_statistics(df):
    """
    Print summary statistics to console.

    Args:
        df: Dataframe with results
    """
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY STATISTICS")
    print("=" * 60)

    # Overall statistics
    print(f"\nTotal runs: {len(df)}")
    print(f"Overall success rate: {df['completed'].mean():.1%}")
    print(f"Average progress: {df['progress'].mean():.1%}")

    # By agent type
    print("\n--- By Agent Type ---")
    for agent_type in sorted(df['agent_type'].unique()):
        agent_df = df[df['agent_type'] == agent_type]
        completed_agent = agent_df[agent_df['completed'] == True]

        print(f"\n{agent_type.upper()}:")
        print(f"  Success rate: {agent_df['completed'].mean():.1%}")
        print(f"  Avg progress: {agent_df['progress'].mean():.1%}")
        if not completed_agent.empty:
            print(
                f"  Avg completion time: {completed_agent['time'].mean():.1f}s (±{completed_agent['time'].std():.1f})")
        print(f"  Avg total reward: {agent_df['total_reward'].mean():.1f}")
        print(f"  Avg collisions: {agent_df['collisions'].mean():.1f}")

    # By number of drones
    print("\n--- By Number of Drones ---")
    for num_drones in sorted(df['drones'].unique()):
        drone_df = df[df['drones'] == num_drones]
        completed_drone = drone_df[drone_df['completed'] == True]

        print(f"\n{num_drones} Drone(s):")
        print(f"  Success rate: {drone_df['completed'].mean():.1%}")
        print(f"  Avg progress: {drone_df['progress'].mean():.1%}")
        if not completed_drone.empty:
            print(f"  Avg completion time: {completed_drone['time'].mean():.1f}s")
        print(f"  Avg reward per agent: {drone_df['avg_reward'].mean():.1f}")

    # Best configurations
    print("\n--- Best Configurations ---")
    config_summary = df.groupby(['agent_type', 'drones']).agg({
        'completed': 'mean',
        'time': lambda x: x[df.loc[x.index, 'completed'] == True].mean(),
        'progress': 'mean',
        'avg_reward': 'mean'
    }).round(2)

    # Best success rate
    best_success = config_summary['completed'].idxmax()
    print(f"Best success rate: {best_success} ({config_summary.loc[best_success, 'completed']:.1%})")

    # Fastest completion (among successful)
    valid_times = config_summary['time'].dropna()
    if not valid_times.empty:
        fastest = valid_times.idxmin()
        print(f"Fastest completion: {fastest} ({config_summary.loc[fastest, 'time']:.1f}s)")

    # Best reward
    best_reward = config_summary['avg_reward'].idxmax()
    print(f"Best avg reward: {best_reward} ({config_summary.loc[best_reward, 'avg_reward']:.1f})")

    print("\n" + "=" * 60)


def main():
    """
    Main analysis function.
    """
    args = parse_args()

    # Setup paths
    log_dir = Path(args.log_dir)
    csv_path = log_dir / args.csv_name

    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading data from: {csv_path}")

    # Load and clean data
    df, success_rates = load_and_clean_data(csv_path)

    # Filter by agent types if specified
    if args.agent_types:
        df = df[df['agent_type'].isin(args.agent_types)]
        success_rates = success_rates[success_rates['agent_type'].isin(args.agent_types)]

    if df.empty:
        print("No data to analyze after filtering")
        return

    print(f"Loaded {len(df)} benchmark runs")
    print(f"Agent types: {', '.join(df['agent_type'].unique())}")
    print(f"Drone counts: {', '.join(map(str, sorted(df['drones'].unique())))}")

    # Print summary statistics
    print_summary_statistics(df)

    # Create visualizations
    output_path = log_dir / args.output_name
    create_comprehensive_analysis(df, success_rates, output_path, args.show_plot)

    # Create comparison plot if we have multiple configurations
    if len(df['agent_type'].unique()) > 1 or len(df['drones'].unique()) > 1:
        create_comparison_plot(df, output_path, args.show_plot)

    print(f"\nAnalysis complete! Check the logs directory for output files.")


if __name__ == "__main__":
    main()