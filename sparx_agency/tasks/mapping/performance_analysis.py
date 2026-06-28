from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


data_dir = Path.home() / 'Documents' / 'depth_validator_csv'

phase_a_files_list = list(data_dir.glob("static*"))
files = [str(f) for f in phase_a_files_list]


# Read and combine dataframes
all_dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    df['run'] = i + 1
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# Basic statistics
stats = df_all.describe()
print(stats)

# Let's check mean per run
run_stats = df_all.groupby('run').agg({'mae': 'mean', 'rmse': 'mean', 'temporal_drift': 'mean'}).reset_index()
print("\nStats per Run:")
print(run_stats)

# Plot 1: MAE over time for each run
plt.figure(figsize=(12, 6))
for run in df_all['run'].unique():
    subset = df_all[df_all['run'] == run]
    # Normalize time for plotting overlapping
    plt.plot(subset['ts'] - subset['ts'].min(), subset['mae'], label=f'Run {run}')
plt.title('MAE vs Time (Phase A - Static)')
plt.xlabel('Time (s)')
plt.ylabel('MAE (m)')
plt.legend()
plt.grid(True)
plt.savefig('phase_a_mae_over_time.png')

# Plot 2: Temporal Drift (Jitter)
plt.figure(figsize=(12, 6))
for run in df_all['run'].unique():
    subset = df_all[df_all['run'] == run]
    plt.plot(subset['ts'] - subset['ts'].min(), subset['temporal_drift'], label=f'Run {run}')
plt.title('Temporal Drift (Jitter) vs Time (Phase A - Static)')
plt.xlabel('Time (s)')
plt.ylabel('Jitter (m)')
plt.legend()
plt.grid(True)
plt.savefig('phase_a_jitter_over_time.png')

# Plot 3: Boxplot of MAE and Temporal Drift
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x='run', y='mae')
plt.title('Distribution of MAE per Run')
plt.savefig('phase_a_mae_boxplot.png')

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x='run', y='temporal_drift')
plt.title('Distribution of Temporal Drift (Jitter) per Run')
plt.savefig('phase_a_jitter_boxplot.png')


# 1. Setup and File Loading
phase_a_files_list = list(data_dir.glob("static*"))
files = [str(f) for f in phase_a_files_list]

clean_dfs = []

# 2. Data Cleaning: Detecting the Crash
# Drones in Gazebo show extreme roll/pitch (>20 deg/rad) when they flip or fall.
for i, f in enumerate(files):
    df = pd.read_csv(f)

    # Identify the first row where the drone is no longer level (the crash)
    crash_indices = df[(df['roll'].abs() > 20) | (df['pitch'].abs() > 20)].index

    if len(crash_indices) > 0:
        cut_point = crash_indices[0]
        df_clean = df.iloc[:cut_point].copy()  # Keep only data BEFORE the fall
        print(f"Run {i + 1}: Cut at index {cut_point} (Crash detected)")
    else:
        df_clean = df.copy()

    df_clean['run'] = i + 1
    clean_dfs.append(df_clean)

# Combine all valid flight data
df_phase_b = pd.concat(clean_dfs, ignore_index=True)

# 3. Graph 1: MAE over Time (Stability)
plt.figure(figsize=(12, 6))
for run in df_phase_b['run'].unique():
    subset = df_phase_b[df_phase_b['run'] == run]
    # Normalize timestamp to start at 0 for comparison
    plt.plot(subset['ts'] - subset['ts'].min(), subset['mae'], label=f'Run {run}')

plt.title('MAE vs Flight Time (Phase B - Dynamic)')
plt.xlabel('Flight Duration (s)')
plt.ylabel('MAE (m)')
plt.legend()
plt.grid(True)
plt.savefig('phase_b_mae_v_time.png')

# 4. Graph 2: Accuracy vs. Pitch (Wide-Angle Distortion Check)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_phase_b, x='pitch', y='mae', hue='run', alpha=0.3)
plt.title('Effect of Drone Pitch on Depth Accuracy (MAE)')
plt.xlabel('Pitch Angle')
plt.ylabel('MAE (m)')
plt.savefig('phase_b_mae_vs_pitch.png')

# 5. Graph 3: Temporal Jitter (Movement Noise)
plt.figure(figsize=(12, 6))
for run in df_phase_b['run'].unique():
    subset = df_phase_b[df_phase_b['run'] == run]
    plt.plot(subset['ts'] - subset['ts'].min(), subset['temporal_drift'], label=f'Run {run}', alpha=0.7)

plt.title('Temporal Jitter during Movement (Phase B)')
plt.xlabel('Time (s)')
plt.ylabel('Jitter (m)')
plt.legend()
plt.grid(True)
plt.savefig('phase_b_jitter_v_time.png')