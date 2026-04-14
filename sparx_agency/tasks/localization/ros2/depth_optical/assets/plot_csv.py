import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_residuals(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))

    plt.plot(df['Frame'], df['Center_Err_du'], marker='o', label='Center Pixel Error (du)', color='#1f77b4', linewidth=2)
    plt.plot(df['Frame'], df['Right_Err_du'], marker='x', label='Right Edge Pixel Error (du)', color='#d62728', linewidth=2, linestyle='--')

    plt.title('Optical Flow Prediction Error (Residuals) Over Time', fontsize=16, pad=15)
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Absolute Error (pixels/sec)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_residuals("/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/csv_eval/csv_results/residuals_log.csv")