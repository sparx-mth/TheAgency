import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_velocity_zones_with_gt(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))

    plt.plot(df['Frame'], df['GT_Vx'], label='Ground Truth (Actual Velocity)', color='black', linewidth=3, zorder=5)

    plt.plot(df['Frame'], df['Center_Vx'], marker='o', label='Center Zone (Estimated)', color='#2ca02c', linewidth=2, alpha=0.8)
    plt.plot(df['Frame'], df['Global_Vx'], marker='s', label='Global Velocity (Estimated)', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(df['Frame'], df['Right_Vx'], marker='x', label='Right Edge (Noisy)', color='#d62728', linewidth=1.5, linestyle='--', alpha=0.5)

    # Highlight zones with vertical bands
    plt.title('Velocity Estimation vs. Ground Truth (Forward Vx)', fontsize=16, pad=15)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Forward Velocity (m/s)', fontsize=12)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_velocity_zones_with_gt("/home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv")