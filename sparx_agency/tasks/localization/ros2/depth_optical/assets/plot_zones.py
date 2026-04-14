import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_velocity_zones(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))

    plt.plot(df['Frame'], df['Center_Vx'], marker='o', label='Center Zone Velocity (Stable)', color='#2ca02c', linewidth=2)
    plt.plot(df['Frame'], df['Global_Vx'], marker='s', label='Global Velocity (All points)', color='#1f77b4', linewidth=2, linestyle='-')
    plt.plot(df['Frame'], df['Right_Vx'], marker='x', label='Right Edge Velocity (Noisy)', color='#d62728', linewidth=1.5, linestyle='--')

    plt.title('Forward Velocity (Vx) Estimation: Center vs. Edge vs. Global', fontsize=16, pad=15)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Forward Velocity (m/s)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_velocity_zones("/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/zone_velocities_log.csv")
