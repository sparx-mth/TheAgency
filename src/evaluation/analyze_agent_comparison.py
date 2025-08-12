"""
analyze_agent_comparison.py

Enhanced analysis script for comparing agent performance with advanced visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def load_results(csv_path="results/agent_comparison_results.csv"):
    """Load results from CSV file."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found at {csv_path}. Run compare_agents.py first!")

    df = pd.read_csv(csv_path)
    return df


def create_performance_heatmap(df, save_path="results/performance_heatmap.png"):
    """
    Create a heatmap showing performance across multiple metrics.

    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    # Calculate normalized metrics for each agent
    metrics = {}

    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]

        metrics[agent_type] = {
            'Success Rate': agent_df['completed'].mean(),
            'Progress': agent_df['final_progress'].mean(),
            'Reward': agent_df['total_reward'].mean() / df['total_reward'].max(),  # Normalize
            'Efficiency': 1 - (agent_df['steps'].mean() / df['steps'].max()),  # Inverse normalized
            'Discovery Rate': agent_df['discovery_rate'].mean() / df['discovery_rate'].max(),  # Normalize
            'Low Collisions': 1 - (
                agent_df['collisions'].mean() / df['collisions'].max() if df['collisions'].max() > 0 else 0),
        }

    # Create heatmap data
    heatmap_data = pd.DataFrame(metrics).T

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Score'})

    ax.set_title('Agent Performance Heatmap\n(All metrics normalized to 0-1)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Performance Metrics', fontsize=12)
    ax.set_ylabel('Agent Type', fontsize=12)

    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')

    # Make agent names uppercase
    ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Performance heatmap saved to {save_path}")

    return fig


def create_radar_chart(df, save_path="results/radar_comparison.png"):
    """
    Create radar chart comparing agents across multiple dimensions.

    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    # Calculate metrics for each agent
    metrics_dict = {}

    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]

        # Calculate and normalize metrics (0-1 scale)
        success_rate = agent_df['completed'].mean()
        avg_progress = agent_df['final_progress'].mean()
        avg_reward = (agent_df['total_reward'].mean() - df['total_reward'].min()) / (
                    df['total_reward'].max() - df['total_reward'].min())
        efficiency = 1 - (agent_df['steps'].mean() - df['steps'].min()) / (df['steps'].max() - df['steps'].min())
        discovery_rate = (agent_df['discovery_rate'].mean() - df['discovery_rate'].min()) / (
                    df['discovery_rate'].max() - df['discovery_rate'].min())
        low_collisions = 1 - (agent_df['collisions'].mean() - df['collisions'].min()) / (
                    df['collisions'].max() - df['collisions'].min() + 0.001)

        metrics_dict[agent_type] = [success_rate, avg_progress, avg_reward,
                                    efficiency, discovery_rate, low_collisions]

    # Set up radar chart
    categories = ['Success\nRate', 'Avg\nProgress', 'Reward\n(norm)',
                  'Efficiency', 'Discovery\nRate', 'Low\nCollisions']
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Colors for each agent
    colors = {'random': '#FF6B6B', 'frontier': '#4ECDC4', 'ppo': '#45B7D1'}

    # Plot data for each agent
    for agent_type, values in metrics_dict.items():
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=agent_type.upper(),
                color=colors.get(agent_type, 'gray'))
        ax.fill(angles, values, alpha=0.25, color=colors.get(agent_type, 'gray'))

    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)

    # Add grid
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.grid(True)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Add title
    plt.title('Multi-Dimensional Agent Comparison\n(Radar Chart)',
              size=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Radar chart saved to {save_path}")

    return fig


def create_statistical_analysis(df, save_path="results/statistical_analysis.png"):
    """
    Perform and visualize statistical analysis of agent performance.

    Args:
        df: DataFrame with results
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'random': '#FF6B6B', 'frontier': '#4ECDC4', 'ppo': '#45B7D1'}

    # 1. Confidence Intervals for Success Rate
    ax = axes[0, 0]
    for i, agent_type in enumerate(df['agent_type'].unique()):
        agent_df = df[df['agent_type'] == agent_type]
        success_rate = agent_df['completed'].mean()
        ci = stats.bootstrap((agent_df['completed'].values,), np.mean,
                             confidence_level=0.95, n_resamples=1000, method='percentile')

        ax.errorbar(i, success_rate,
                    yerr=[[success_rate - ci.confidence_interval.low],
                          [ci.confidence_interval.high - success_rate]],
                    fmt='o', capsize=5, capthick=2, markersize=10,
                    color=colors[agent_type], label=agent_type.upper())

    ax.set_xticks(range(len(df['agent_type'].unique())))
    ax.set_xticklabels([a.upper() for a in df['agent_type'].unique()])
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate with 95% CI', fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3)

    # 2. Distribution Comparison - Steps
    ax = axes[0, 1]
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        ax.hist(agent_df['steps'], alpha=0.5, label=agent_type.upper(),
                color=colors[agent_type], bins=15)

    ax.set_xlabel('Steps')
    ax.set_ylabel('Frequency')
    ax.set_title('Steps Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Reward vs Progress Scatter
    ax = axes[0, 2]
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        ax.scatter(agent_df['final_progress'] * 100, agent_df['total_reward'],
                   label=agent_type.upper(), color=colors[agent_type],
                   s=100, alpha=0.7, edgecolors='black', linewidth=1)

    ax.set_xlabel('Final Progress (%)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward vs Progress', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Box Plot Comparison - Reward per Step
    ax = axes[1, 0]
    box_data = [df[df['agent_type'] == agent]['avg_reward_per_step'].values
                for agent in df['agent_type'].unique()]
    bp = ax.boxplot(box_data, labels=[a.upper() for a in df['agent_type'].unique()],
                    patch_artist=True)

    for patch, agent in zip(bp['boxes'], df['agent_type'].unique()):
        patch.set_facecolor(colors[agent])
        patch.set_alpha(0.7)

    ax.set_ylabel('Reward per Step')
    ax.set_title('Reward Efficiency Comparison', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. Cumulative Success Over Trials
    ax = axes[1, 1]
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type].sort_values('trial')
        cumsum = agent_df['completed'].cumsum()
        ax.plot(agent_df['trial'], cumsum, marker='o',
                label=agent_type.upper(), color=colors[agent_type],
                linewidth=2, markersize=6)

    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Cumulative Successes')
    ax.set_title('Cumulative Success Count', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Performance Stability (Standard Deviation)
    ax = axes[1, 2]
    stability_data = []
    for agent_type in df['agent_type'].unique():
        agent_df = df[df['agent_type'] == agent_type]
        stability_data.append({
            'Agent': agent_type.upper(),
            'Progress STD': agent_df['final_progress'].std(),
            'Reward STD': agent_df['total_reward'].std() / agent_df['total_reward'].mean(),
            'Steps STD': agent_df['steps'].std() / agent_df['steps'].mean()
        })

    stability_df = pd.DataFrame(stability_data)
    x = np.arange(len(stability_df))
    width = 0.25

    ax.bar(x - width, stability_df['Progress STD'], width, label='Progress',
           color='skyblue')
    ax.bar(x, stability_df['Reward STD'], width, label='Reward (CV)',
           color='lightcoral')
    ax.bar(x + width, stability_df['Steps STD'], width, label='Steps (CV)',
           color='lightgreen')

    ax.set_xticks(x)
    ax.set_xticklabels(stability_df['Agent'])
    ax.set_ylabel('Variability')
    ax.set_title('Performance Stability\n(Lower is More Stable)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Statistical Analysis of Agent Performance',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Statistical analysis saved to {save_path}")

    return fig


def perform_significance_tests(df):
    """
    Perform statistical significance tests between agents.

    Args:
        df: DataFrame with results
    """
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    agents = df['agent_type'].unique()

    # Test for each metric
    metrics = ['completed', 'final_progress', 'total_reward', 'steps', 'collisions']

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 40)

        # Perform ANOVA or Kruskal-Wallis test
        groups = [df[df['agent_type'] == agent][metric].values for agent in agents]

        # Check normality
        normal = all(stats.shapiro(group)[1] > 0.05 for group in groups if len(group) >= 3)

        if normal:
            # Use ANOVA
            statistic, p_value = stats.f_oneway(*groups)
            test_name = "ANOVA"
        else:
            # Use Kruskal-Wallis (non-parametric)
            statistic, p_value = stats.kruskal(*groups)
            test_name = "Kruskal-Wallis"

        print(f"  {test_name} Test: statistic={statistic:.3f}, p-value={p_value:.4f}")

        if p_value < 0.05:
            print("  Result: Significant difference between agents (p < 0.05)")

            # Pairwise comparisons
            print("\n  Pairwise Comparisons (Mann-Whitney U):")
            for i, agent1 in enumerate(agents):
                for agent2 in agents[i + 1:]:
                    group1 = df[df['agent_type'] == agent1][metric].values
                    group2 = df[df['agent_type'] == agent2][metric].values

                    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

                    # Apply Bonferroni correction
                    corrected_p = p_value * 3  # 3 comparisons

                    significance = "***" if corrected_p < 0.001 else "**" if corrected_p < 0.01 else "*" if corrected_p < 0.05 else "ns"

                    print(
                        f"    {agent1.upper()} vs {agent2.upper()}: p={p_value:.4f} (corrected={corrected_p:.4f}) {significance}")
        else:
            print("  Result: No significant difference between agents (p >= 0.05)")

    print("\n" + "=" * 70)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("P-values are Bonferroni corrected for multiple comparisons")


def create_summary_report(df, save_path="results/summary_report.txt"):
    """
    Create a comprehensive text summary report.

    Args:
        df: DataFrame with results
        save_path: Path to save report
    """
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SLAM AGENT COMPARISON - COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("EXPERIMENT CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Map: house_map_10.txt (10x10 grid)\n")
        f.write(f"Trials per agent: {df.groupby('agent_type').size().iloc[0]}\n")
        f.write(f"Camera configuration: range=5, FOV=90°, rays=20\n")
        f.write(f"Max steps: 500\n")
        f.write(f"Agents tested: {', '.join([a.upper() for a in df['agent_type'].unique()])}\n\n")

        f.write("PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")

        # Summary for each agent
        for agent_type in df['agent_type'].unique():
            agent_df = df[df['agent_type'] == agent_type]
            completed_df = agent_df[agent_df['completed'] == True]

            f.write(f"\n{agent_type.upper()} AGENT:\n")
            f.write(
                f"  • Success Rate: {agent_df['completed'].mean():.1%} ({completed_df.shape[0]}/{agent_df.shape[0]} trials)\n")
            f.write(
                f"  • Average Progress: {agent_df['final_progress'].mean():.1%} ± {agent_df['final_progress'].std():.1%}\n")
            f.write(f"  • Average Steps: {agent_df['steps'].mean():.0f} ± {agent_df['steps'].std():.0f}\n")
            f.write(
                f"  • Average Reward: {agent_df['total_reward'].mean():.2f} ± {agent_df['total_reward'].std():.2f}\n")

            if not completed_df.empty:
                f.write(
                    f"  • Completion Time: {completed_df['time'].mean():.2f}s ± {completed_df['time'].std():.2f}s\n")

            f.write(
                f"  • Average Collisions: {agent_df['collisions'].mean():.1f} ± {agent_df['collisions'].std():.1f}\n")
            f.write(f"  • Discovery Rate: {agent_df['discovery_rate'].mean():.3f} cells/step\n")
            f.write(f"  • Reward Efficiency: {agent_df['avg_reward_per_step'].mean():.3f} reward/step\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("RANKINGS\n")
        f.write("-" * 40 + "\n")

        # Rank agents by different metrics
        rankings = {
            'Success Rate': df.groupby('agent_type')['completed'].mean().sort_values(ascending=False),
            'Average Progress': df.groupby('agent_type')['final_progress'].mean().sort_values(ascending=False),
            'Average Reward': df.groupby('agent_type')['total_reward'].mean().sort_values(ascending=False),
            'Fewest Steps': df.groupby('agent_type')['steps'].mean().sort_values(ascending=True),
            'Discovery Rate': df.groupby('agent_type')['discovery_rate'].mean().sort_values(ascending=False),
            'Fewest Collisions': df.groupby('agent_type')['collisions'].mean().sort_values(ascending=True),
        }

        for metric, ranking in rankings.items():
            f.write(f"\n{metric}:\n")
            for i, (agent, value) in enumerate(ranking.items(), 1):
                if 'Rate' in metric or 'Progress' in metric or 'Success' in metric:
                    f.write(f"  {i}. {agent.upper()}: {value:.1%}\n")
                elif metric in ['Fewest Steps', 'Fewest Collisions']:
                    f.write(f"  {i}. {agent.upper()}: {value:.1f}\n")
                else:
                    f.write(f"  {i}. {agent.upper()}: {value:.2f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS\n")
        f.write("-" * 40 + "\n")

        # Determine overall winner
        points = {agent: 0 for agent in df['agent_type'].unique()}

        for metric, ranking in rankings.items():
            for i, agent in enumerate(ranking.index):
                points[agent] += (3 - i)  # 3 points for 1st, 2 for 2nd, 1 for 3rd

        winner = max(points, key=points.get)

        f.write(f"\nOverall Performance Winner: {winner.upper()}\n")
        f.write(f"(Based on combined rankings across all metrics)\n\n")

        f.write("Point Breakdown:\n")
        for agent, pts in sorted(points.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  • {agent.upper()}: {pts} points\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"Summary report saved to {save_path}")


def main():
    """Main analysis function."""
    print("\n" + "=" * 70)
    print("ENHANCED AGENT COMPARISON ANALYSIS")
    print("=" * 70)

    # Load results
    try:
        df = load_results()
        print(f"\nLoaded {len(df)} trial results")
        print(f"Agents: {', '.join([a.upper() for a in df['agent_type'].unique()])}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run compare_agents.py first to generate results.")
        return

    # Create all visualizations
    print("\nGenerating visualizations...")

    create_performance_heatmap(df)
    create_radar_chart(df)
    create_statistical_analysis(df)

    # Perform statistical tests
    perform_significance_tests(df)

    # Create summary report
    create_summary_report(df)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("Check the 'results/' directory for all outputs:")
    print("  • performance_heatmap.png")
    print("  • radar_comparison.png")
    print("  • statistical_analysis.png")
    print("  • summary_report.txt")
    print("=" * 70)

    # Show plots
    plt.show()


if __name__ == "__main__":
    main()