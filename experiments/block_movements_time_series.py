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
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    FONT = 'Sans Serif'
    OPACITY = 0.7
    print("Failed to import colors")

def load_block_movement_data(file_path):
    """
    Load block movement data from a file.
    Expected format: CSV or JSON with columns/fields for time and block movements between different tiers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Block movement data file {file_path} not found")
    
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        try:
            movement_df = pd.read_csv(file_path)
            print(f"Loaded {len(movement_df)} records from {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load block movement data: {e}")
    elif file_path.endswith('.json') or file_path.endswith('.jsonl'):
        try:
            movement_df = pd.read_json(file_path, lines=file_path.endswith('.jsonl'))
            print(f"Loaded {len(movement_df)} records from {file_path}")
        except Exception as e:
            raise Exception(f"Failed to load block movement data: {e}")
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use CSV or JSON/JSONL.")
    
    return movement_df

def plot_block_movements_time_series(data_file, output_file=None, title="Cache Block Movements Time Series"):
    """
    Create a time series plot of block movements between different memory tiers.
    
    Parameters:
    - data_file: Path to the data file containing block movement data
    - output_file: Path to save the output plot (if None, will use default name)
    - title: Title for the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Load the block movement data
    movement_df = load_block_movement_data(data_file)
    
    # Ensure the required columns exist
    required_columns = [
        'time_seconds', 
        'CPU_TO_GPU', 
        'CPU_TO_NVME', 
        'GPU_TO_CPU', 
        'GPU_TO_NVME', 
        'NVME_TO_CPU', 
        'NVME_TO_GPU'
    ]
    
    missing_columns = [col for col in required_columns if col not in movement_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in data file: {missing_columns}")
    
    # Create the plot
    plt.rcParams.update({'font.size': 12, 'font.family': FONT})
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Ensure the plot starts from 0 on x-axis
    min_time = 0
    
    # Make sure to include the point at x=0 for each line
    if movement_df['time_seconds'].min() > 0:
        # Add a point at x=0 for each line to ensure they start from the origin
        zero_point = pd.DataFrame({
            'time_seconds': [0],
            'CPU_TO_GPU': [0],
            'CPU_TO_NVME': [0],
            'GPU_TO_CPU': [0],
            'GPU_TO_NVME': [0],
            'NVME_TO_CPU': [0],
            'NVME_TO_GPU': [0]
        })
        plot_df = pd.concat([zero_point, movement_df]).reset_index(drop=True)
    else:
        plot_df = movement_df
    
    # Plot each movement type with its own color
    movement_types = [
        ('CPU_TO_GPU', 'CPU → GPU', COLORS[0]),
        ('CPU_TO_NVME', 'CPU → NVMe', COLORS[1]),
        ('GPU_TO_CPU', 'GPU → CPU', COLORS[2]),
        ('GPU_TO_NVME', 'GPU → NVMe', COLORS[3]),
        ('NVME_TO_CPU', 'NVMe → CPU', COLORS[4]),
        ('NVME_TO_GPU', 'NVMe → GPU', COLORS[5] if len(COLORS) > 5 else '#8c564b')
    ]
    
    for col, label, color in movement_types:
        ax.plot(plot_df['time_seconds'], plot_df[col], 
                color=color, linewidth=2, label=label)
    
    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of Blocks Moved')
    ax.set_title(title)
    
    # Set axis limits
    ax.set_xlim(0, plot_df['time_seconds'].max())
    
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
        output_file = 'output/block_movements_time_series.png'
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()

def generate_sample_data(output_file, num_points=100, time_range=(0, 3600)):
    """
    Generate sample block movement data for testing the plot.
    
    Parameters:
    - output_file: Path to save the generated data
    - num_points: Number of data points to generate
    - time_range: Range of time in seconds (start, end)
    """
    # Create time points
    time_points = np.linspace(time_range[0], time_range[1], num_points)
    
    # Generate sample block movements with some randomness and trends
    # CPU to GPU: steady increase
    cpu_to_gpu = 100 + 300 * (time_points / time_range[1]) + np.random.normal(0, 50, num_points)
    cpu_to_gpu = np.clip(cpu_to_gpu, 0, None)
    
    # CPU to NVMe: periodic pattern
    cpu_to_nvme = 50 + 100 * np.sin(time_points / 500) + np.random.normal(0, 20, num_points)
    cpu_to_nvme = np.clip(cpu_to_nvme, 0, None)
    
    # GPU to CPU: spiky pattern
    gpu_to_cpu = 200 + 150 * np.sin(time_points / 200) + np.random.normal(0, 80, num_points)
    gpu_to_cpu = np.clip(gpu_to_cpu, 0, None)
    
    # GPU to NVMe: low activity
    gpu_to_nvme = 30 + 20 * np.sin(time_points / 300) + np.random.normal(0, 10, num_points)
    gpu_to_nvme = np.clip(gpu_to_nvme, 0, None)
    
    # NVMe to CPU: increasing over time
    nvme_to_cpu = 50 + 200 * (time_points / time_range[1])**2 + np.random.normal(0, 30, num_points)
    nvme_to_cpu = np.clip(nvme_to_cpu, 0, None)
    
    # NVMe to GPU: decreasing over time
    nvme_to_gpu = 250 - 100 * (time_points / time_range[1]) + np.random.normal(0, 40, num_points)
    nvme_to_gpu = np.clip(nvme_to_gpu, 0, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_seconds': time_points,
        'CPU_TO_GPU': cpu_to_gpu,
        'CPU_TO_NVME': cpu_to_nvme,
        'GPU_TO_CPU': gpu_to_cpu,
        'GPU_TO_NVME': gpu_to_nvme,
        'NVME_TO_CPU': nvme_to_cpu,
        'NVME_TO_GPU': nvme_to_gpu
    })
    
    # Convert to integers as we're counting blocks
    for col in df.columns:
        if col != 'time_seconds':
            df[col] = df[col].astype(int)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data generated and saved to {output_file}")
    
    return df

if __name__ == "__main__":
    # Check if a data file is provided as an argument
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        plot_block_movements_time_series(data_file)
    else:
        # Generate and use sample data
        print("No data file provided. Generating sample data...")
        sample_data_file = 'output/sample_block_movements_data.csv'
        generate_sample_data(sample_data_file)
        plot_block_movements_time_series(sample_data_file, title="Sample Cache Block Movements Time Series")
