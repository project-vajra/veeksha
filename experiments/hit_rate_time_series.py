import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path to import constants
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from constants import *
    print("Successfully imported constants")
except ImportError:
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    FONT = 'Sans Serif'
    OPACITY = 0.7
    print("Failed to import colors")

def load_hit_rate_data(file_path):
    """
    Load cache hit rate data from a file.
    Expected format: CSV or JSON with columns/fields for time and hit rates for each tier.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hit rate data file {file_path} not found")
    
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        try:
            hit_rate_df = pd.read_csv(file_path)
            print(f"Loaded {len(hit_rate_df)} records from {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load hit rate data: {e}")
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        try:
            hit_rate_df = pd.read_json(file_path, lines=file_path.endswith('.jsonl'))
            print(f"Loaded {len(hit_rate_df)} records from {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load hit rate data: {e}")
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use CSV or JSON/JSONL.")
    
    return hit_rate_df

def plot_hit_rate_time_series(data_file, output_file=None, title="Cache Hit Rate Time Series"):
    """
    Create a time series plot of hit rates for HBM, DRAM, and NVMe cache tiers.
    
    Parameters:
    - data_file: Path to the data file containing hit rate data
    - output_file: Path to save the output plot (if None, will use default name)
    - title: Title for the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Load the hit rate data
    hit_rate_df = load_hit_rate_data(data_file)
    
    # Ensure the required columns exist
    required_columns = ['time_seconds', 'hbm_hit_rate', 'dram_hit_rate', 'nvme_hit_rate']
    missing_columns = [col for col in required_columns if col not in hit_rate_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in data file: {missing_columns}")
    
    # Create the plot
    plt.rcParams.update({'font.size': 12, 'font.family': FONT})
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ensure the plot starts from 0 on x-axis
    min_time = 0
    
    # Plot each cache tier with its own color
    # Make sure to include the point at x=0 for each line
    if hit_rate_df['time_seconds'].min() > 0:
        # Add a point at x=0 for each line to ensure they start from the origin
        zero_point = pd.DataFrame({
            'time_seconds': [0],
            'hbm_hit_rate': [0],
            'dram_hit_rate': [0],
            'nvme_hit_rate': [0]
        })
        plot_df = pd.concat([zero_point, hit_rate_df]).reset_index(drop=True)
    else:
        plot_df = hit_rate_df
    
    ax.plot(plot_df['time_seconds'], plot_df['hbm_hit_rate'], 
            color=COLORS[0], linewidth=2, label='HBM')
    ax.plot(plot_df['time_seconds'], plot_df['dram_hit_rate'], 
            color=COLORS[1], linewidth=2, label='DRAM')
    ax.plot(plot_df['time_seconds'], plot_df['nvme_hit_rate'], 
            color=COLORS[2], linewidth=2, label='NVMe')
    
    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(0, plot_df['time_seconds'].max())
    ax.set_ylim(0, 100)
    
    # Remove padding between plot area and axes
    plt.margins(x=0)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend in the top right corner
    ax.legend(loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if output_file is provided
    if output_file is None:
        output_file = 'output/hit_rate_time_series.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def generate_sample_data(output_file, num_points=100, time_range=(0, 3600)):
    """
    Generate sample hit rate data for testing the plot.
    
    Parameters:
    - output_file: Path to save the generated data
    - num_points: Number of data points to generate
    - time_range: Range of time in seconds (start, end)
    """
    # Create time points
    time_points = np.linspace(time_range[0], time_range[1], num_points)
    
    # Generate sample hit rates with some randomness and trends
    # HBM: high hit rate that slightly decreases over time
    hbm_hit_rate = 95 - 15 * (time_points / time_range[1]) + np.random.normal(0, 3, num_points)
    hbm_hit_rate = np.clip(hbm_hit_rate, 0, 100)
    
    # DRAM: medium hit rate with some fluctuations
    dram_hit_rate = 70 + 10 * np.sin(time_points / 500) + np.random.normal(0, 5, num_points)
    dram_hit_rate = np.clip(dram_hit_rate, 0, 100)
    
    # NVMe: low hit rate that increases over time
    nvme_hit_rate = 30 + 20 * (time_points / time_range[1]) + np.random.normal(0, 7, num_points)
    nvme_hit_rate = np.clip(nvme_hit_rate, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_seconds': time_points,
        'hbm_hit_rate': hbm_hit_rate,
        'dram_hit_rate': dram_hit_rate,
        'nvme_hit_rate': nvme_hit_rate
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data generated and saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Check if a data file is provided as an argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        plot_hit_rate_time_series(data_file)
    else:
        # Generate and use sample data
        print("No data file provided. Generating sample data...")
        sample_data_file = 'output/sample_hit_rate_data.csv'
        generate_sample_data(sample_data_file)
        plot_hit_rate_time_series(sample_data_file, title="Sample Cache Hit Rate Time Series")
