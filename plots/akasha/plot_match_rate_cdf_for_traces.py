import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from collections import defaultdict

# Import constants from the constants file
from constants import COLORS, OPACITY, FONT


def create_running_hashes(hash_ids):
    """Create running hashes for a sequence of hash IDs.
    Each running hash at position i is a hash of the sequence from 0 to i."""
    if not hash_ids:
        return []
    
    running_hashes = []
    
    # For each position i, create a hash of the tuple of all elements from 0 to i
    for i in range(len(hash_ids)):
        # Create a tuple of the prefix and hash it
        prefix_tuple = tuple(hash_ids[:i+1])
        prefix_hash = hash(prefix_tuple) % (2**32)  # Keep hash size manageable
        running_hashes.append(prefix_hash)
    
    return running_hashes


def calculate_match_percentage(trace_file, block_size=512):
    """Calculate the match percentage for a trace file."""
    data = []
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {trace_file}")
    except FileNotFoundError:
        print(f"Error: Trace file '{trace_file}' not found.")
        return []

    if not data:
        print(f"No valid data loaded from {trace_file}.")
        return []
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)  # Reset index after sorting
    
    # Store the original hash IDs before replacing them
    df['original_hash_ids'] = df['hash_ids']

    # Replace hash_ids with running hashes
    df['hash_ids'] = df.apply(lambda row: create_running_hashes(row['original_hash_ids']), axis=1)

    # Store a map of hash sequences to their lengths
    hash_to_length = {}

    # Initialize columns
    df['prefix_match_blocks'] = 0

    # Process each request in order
    for idx, row in df.iterrows():
        current_hashes = row['hash_ids']
        best_match_len = 0
        
        # Check for matches of increasing length
        for prefix_len in range(1, len(current_hashes) + 1):
            # Create a tuple of the prefix hashes (immutable for dict key)
            prefix = tuple(current_hashes[:prefix_len])
            
            # If this prefix exists in our map, we have a match
            if prefix in hash_to_length:
                best_match_len = prefix_len
        
        # Store the best match length for this request
        df.at[idx, 'prefix_match_blocks'] = best_match_len
        
        # Add all prefixes of the current hash sequence to our map
        for prefix_len in range(1, len(current_hashes) + 1):
            prefix = tuple(current_hashes[:prefix_len])
            hash_to_length[prefix] = prefix_len

    # Calculate matched tokens (approx)
    df['prefix_match_tokens'] = df['prefix_match_blocks'] * block_size
    
    # Calculate match percentage for each request
    df['match_percentage'] = (df['prefix_match_tokens'] / df['input_length']) * 100
    df['match_percentage'] = df['match_percentage'].fillna(0)  # Replace NaN with 0
    
    return df['match_percentage'].tolist()


def load_and_process_data(trace_type):
    """Load and process data for a specific trace type."""
    FILE_PATH = f'../../data/processed_traces/{trace_type}_trace.jsonl'
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return []

    # Calculate match percentage for each trace file and combine results
    all_match_percentages = calculate_match_percentage(FILE_PATH)
    
    return all_match_percentages


def plot_match_rate_cdf(trace_types):
    """Plot the CDF of match length percentage for different trace types in a single graph."""
    # Set up the figure with appropriate size for a single column in SOSP paper
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': FONT})

    # Create a figure optimized for single column in SOSP paper (typically ~3.33 inches wide)
    fig, ax = plt.subplots(figsize=(3.33, 2.5))

    # Create a dictionary to store results for each trace type
    results = {}

    # Load data for each trace type
    for i, trace_type in enumerate(trace_types):
        match_percentages = load_and_process_data(trace_type)
        if match_percentages:
            results[trace_type] = match_percentages

    # Plot the data for each trace type
    for i, (trace_type, match_percentages) in enumerate(results.items()):
        # Sort the values for CDF plotting
        plotting_values = np.sort(match_percentages)
        y = np.linspace(0, 1, len(plotting_values))
        
        # Use different colors for different trace types
        line, = ax.plot(plotting_values, y, color=COLORS[i], 
                        label=trace_type.capitalize(), linewidth=2, alpha=OPACITY)
    
    # Configure the plot
    ax.grid(which='major', axis='both', linestyle='--', alpha=0.7)
    # set axis labels to be bold
    ax.set_xlabel("Match Percentage (%)", fontsize=10, fontweight='bold')
    ax.set_ylabel("CDF", fontsize=10, fontweight='bold')
    # ax.set_xscale('log')
    # ax.set_xticks([0.1, 1, 10, 100])
    ax.set_xlim([0, 100])
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    
    # Add legend with smaller font size
    legend = ax.legend(loc='lower right', fontsize=8)

    # Tight layout to optimize space usage
    plt.tight_layout()

    # Save the figure with high resolution
    plt.savefig('./output/match_rate_cdf_for_traces.pdf', bbox_inches='tight')
    plt.savefig('./output/match_rate_cdf_for_traces.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Define trace types to analyze
    trace_types = ["conversation", "toolagent"]
    
    # Plot the CDF of match length percentage
    plot_match_rate_cdf(trace_types)