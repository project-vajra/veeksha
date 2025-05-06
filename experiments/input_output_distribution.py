import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
from pathlib import Path

from constants import COLORS, FONT, OPACITY

# Block size used for hashing, from the MOONCAKE paper
BLOCK_SIZE = 512

def load_trace_data(workload):
    # Path to the trace file
    trace_file = f'../data/{workload}_trace.jsonl'
    
    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"Trace file {trace_file} not found")
    
    # Load the data
    try:
        conversation_trace_df = pd.read_json(trace_file, lines=True)
        print(f"Loaded {len(conversation_trace_df)} records from {trace_file}")
    except Exception as e:
        raise Exception(f"Failed to load trace data: {e}")
    
    return conversation_trace_df


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


log_x = True

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Load the conversation trace data
conversation_df = load_trace_data('conversation')

# Print some statistics about the data
print(f"\nData Statistics:")
print(f"Input tokens - min: {conversation_df['input_length'].min()}, max: {conversation_df['input_length'].max()}, mean: {conversation_df['input_length'].mean():.2f}")
print(f"Output tokens - min: {conversation_df['output_length'].min()}, max: {conversation_df['output_length'].max()}, mean: {conversation_df['output_length'].mean():.2f}")

# Create figure with subplots
plt.rcParams.update({'font.size': 12, 'font.family': FONT})
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Define the workload names and their properties
workloads = ['Conversation', 'Tool&Agent', 'SWEBench']

# Define x-axis limits for each workload
x_limits = {
    'Conversation': 30000,
    'Tool&Agent': 30000,
    'SWEBench': 30000
}

# For all three plots, we'll use the same conversation data
# but with different x-axis limits to simulate different workloads

    
for i, workload in enumerate(workloads):
    ax = axes[i]
    x_limit = x_limits[workload]

    # Load the trace data for the current workload
    if workload == 'Tool&Agent':
        toolagent_df = load_trace_data('toolagent')
        input_length = toolagent_df['input_length']
        output_length = toolagent_df['output_length']
    elif workload == 'SWEBench':
        sweb_df = load_trace_data('swe_agent_short')
        input_length = sweb_df['input_length']
        output_length = sweb_df['output_length']
    else:
        input_length = conversation_df['input_length']
        output_length = conversation_df['output_length']
    
    # Create bins for the histogram
    bins = np.linspace(0, x_limit, 50)
    
    # Plot input token distribution
    ax.hist(input_length, bins=bins, alpha=OPACITY, 
            color=COLORS[0], density=True, label='Input')
    
    # Plot output token distribution
    ax.hist(output_length, bins=bins, alpha=OPACITY, 
            color=COLORS[1], density=True, label='Output')
    
    # Set labels and title
    ax.set_xlabel('# Tokens')
    if i == 0:
        ax.set_ylabel('Density')
    
    # Set x-axis limits
    ax.set_xlim(0, x_limit)
    
    # Set log scale for x-axis if requested
    if log_x:
        ax.set_xscale('log', base=2)
        # Avoid showing 0 on log scale
        ax.set_xlim(16, x_limit)
        
        # Set specific x-ticks for log scale
        log_ticks = [16, 128, 1024, 8192, 65536]  # 16, 128, 1k, 8k, 64k
        ax.set_xticks(log_ticks)
        ax.set_xticklabels(['16', '128', '1k', '8k', '64k'])
    
    # Format y-axis with scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    # Set subplot title with workload name
    ax.set_title(f'({chr(97+i)}) {workload}')
    
    # Add legend only for the first plot
    if i == 0:
        handles, labels = ax.get_legend_handles_labels()

# Add a common legend at the top
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

# Adjust layout
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save the figure
output_filename = 'output/input_output_distribution_log.png' if log_x else 'output/input_output_distribution.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_filename}")

# Show the plot
plt.show()
