import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_depth_data(input_csv, output_csv):
    """
    Reads depth data, applies Temporal Smoothing (Median + EMA) to reduce jitter,
    saves the smoothed data to a new CSV, and generates an analysis plot.
    """
    
    print(f"Loading data from: {input_csv}...")
    # 1. Load the dataset
    df = pd.read_csv(input_csv)
    
    # 2. Clean the data (Drop rows where depth or Ground Truth are missing)
    df = df.dropna(subset=['da3_depth_m', 'gt_depth_geom_m'])
    
    # 3. Sort the data chronologically for each color/marker
    # This is CRITICAL for time-series operations like rolling windows and EMA
    df = df.sort_values(by=['color', 'ts'])
    
    # ==========================================
    # FILTER PARAMETERS 
    # ==========================================
    MEDIAN_WINDOW = 15  # Removes sudden spikes (outliers). Higher = more robust but might cut corners.
    EMA_ALPHA = 0.1   # Smooths the noise. Lower = smoother but adds delay/lag. Higher = more responsive.
    
    print(f"Applying filters... (Median Window: {MEDIAN_WINDOW}, EMA Alpha: {EMA_ALPHA})")
    
    # ==========================================
    # APPLYING THE FILTERS
    # ==========================================
    # Step 4a: Median Filter (Removes Outliers/Spikes)
    # We use groupby('color') to ensure we don't mix frames from different markers
    df['da3_depth_median'] = df.groupby('color')['da3_depth_m'].transform(
        lambda x: x.rolling(window=MEDIAN_WINDOW, min_periods=1, center=False).median()
    )
    
    # Step 4b: Exponential Moving Average (EMA) (Smooths the continuous noise)
    df['da3_depth_smoothed'] = df.groupby('color')['da3_depth_median'].transform(
        lambda x: x.ewm(alpha=EMA_ALPHA, adjust=False).mean()
    )
    
    # ==========================================
    # CALCULATE JITTER METRICS
    # ==========================================
    # Calculate frame-to-frame difference (velocity/jitter)
    df['gt_diff'] = df.groupby('color')['gt_depth_geom_m'].diff()
    df['da3_orig_diff'] = df.groupby('color')['da3_depth_m'].diff()
    df['da3_smoothed_diff'] = df.groupby('color')['da3_depth_smoothed'].diff()
    
    # Print statistics per color
    print("\n--- Jitter Reduction Summary ---")
    colors = df['color'].unique()
    for color in colors:
        color_data = df[df['color'] == color]
        
        orig_jitter = color_data['da3_orig_diff'].abs().mean()
        smooth_jitter = color_data['da3_smoothed_diff'].abs().mean()
        
        if pd.notna(orig_jitter) and orig_jitter > 0:
            reduction = (1 - (smooth_jitter / orig_jitter)) * 100
            print(f"Marker [{color.ljust(6)}]: Original Jitter: {orig_jitter:.4f}m | Smoothed: {smooth_jitter:.4f}m | Reduction: {reduction:.1f}%")
            
    # 5. Save the processed data
    print(f"\nSaving processed data to: {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    # ==========================================
    # PLOTTING THE RESULTS
    # ==========================================
    print("Generating plots...")
    fig, axes = plt.subplots(len(colors), 1, figsize=(12, 4 * len(colors)))
    if len(colors) == 1:
        axes = [axes]
        
    for i, color in enumerate(colors):
        color_data = df[df['color'] == color]
        ax = axes[i]
        
        # We calculate an offset just for visualization purposes so the DA3 plots sit on top of the GT
        # This makes it easier to visually compare the "shape" of the noise vs the GT
        offset = color_data['gt_depth_geom_m'].mean() - color_data['da3_depth_smoothed'].mean()
        
        valid_color = color if color in ['red', 'green', 'blue', 'purple', 'yellow', 'cyan', 'magenta', 'orange', 'black', 'white'] else 'blue'
        
        # Plot Ground Truth
        ax.plot(color_data['ts'], color_data['gt_depth_geom_m'], 
                label='Ground Truth', color='black', linewidth=2, linestyle='--')
        
        # Plot Original DA3 Prediction
        ax.plot(color_data['ts'], color_data['da3_depth_m'] + offset, 
                label='Raw DA3 (Offset Applied)', color=valid_color, alpha=0.3, linewidth=1)
        
        # Plot Smoothed DA3 Prediction
        ax.plot(color_data['ts'], color_data['da3_depth_smoothed'] + offset, 
                label='Smoothed DA3 (Offset Applied)', color='red', linewidth=2)
        
        ax.set_title(f'Depth Smoothing Results - Marker: {color}')
        ax.set_xlabel('Timestamp (s)')
        ax.set_ylabel('Depth (m)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('smoothing_analysis.png')
    print("Plot saved as 'smoothing_analysis.png'.")
    print("Done!")

if __name__ == "__main__":
    # Define input and output filenames
    INPUT_FILE = '/home/user/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/assets/depth_validator_csv/da3_markers_20260412_152728.csv'
    OUTPUT_FILE = '/home/user/GIT/TheAgency/sparx_agency/tasks/localization/ros2/depth_optical/assets/depth_validator_csv/da3_markers_smoothed_output.csv'

    # Run the processing pipeline
    process_depth_data(INPUT_FILE, OUTPUT_FILE)