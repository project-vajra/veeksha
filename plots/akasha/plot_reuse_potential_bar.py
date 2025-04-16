import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from collections import defaultdict

# Import constants from the constants file
from constants import COLORS, OPACITY, FONT, HATCHES


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


def calculate_reuse_potential(trace_file, block_size=512):
    """Calculate the maximum reuse potential for a trace file."""
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
        return 0

    if not data:
        print(f"No valid data loaded from {trace_file}.")
        return 0
    
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
    
    # Calculate overall potential reuse ratio (matched tokens / total input tokens)
    total_input_tokens = df['input_length'].sum()
    total_matched_tokens = df['prefix_match_tokens'].sum()
    reuse_ratio = (total_matched_tokens / total_input_tokens) * 100 if total_input_tokens > 0 else 0
    
    return reuse_ratio


def load_and_process_data(trace_type):
    """Load and process data for a specific trace type."""
    FILE_PATH = f'../../data/processed_traces/{trace_type}_trace.jsonl'
    
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return 0

    # Calculate reuse potential for the trace file
    reuse_potential = calculate_reuse_potential(FILE_PATH)
    
    return reuse_potential


def plot_reuse_potential_bar(trace_types):
    """Plot a bar chart showing the maximum reuse potential for different trace types."""
    # Set up the figure with appropriate size for a single column in SOSP paper
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': FONT})

    # Create a figure optimized for single column in SOSP paper (typically ~3.33 inches wide)
    fig, ax = plt.subplots(figsize=(3.33, 2.5))

    # Create a dictionary to store results for each trace type
    results = {}

    # Load data for each trace type
    for i, trace_type in enumerate(trace_types):
        reuse_potential = load_and_process_data(trace_type)
        if reuse_potential > 0:
            results[trace_type] = reuse_potential

    # Prepare data for plotting
    trace_labels = [t.capitalize() for t in results.keys()]
    reuse_values = list(results.values())
    x_pos = np.arange(len(trace_labels))
    
    # Plot the bars
    bars = ax.bar(x_pos, reuse_values, width=0.6, alpha=OPACITY)
    
    # Color each bar according to its trace type
    for i, bar in enumerate(bars):
        bar.set_color(COLORS[i])
        bar.set_hatch(HATCHES[i])

    # Configure the plot
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel("Trace Type", fontsize=10, fontweight='bold')
    ax.set_ylabel("Reuse Potential (%)", fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(trace_labels)
    ax.set_ylim([0, max(reuse_values) * 1.1])  # Add 10% padding at the top
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add value labels on top of each bar
    for i, v in enumerate(reuse_values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=8)

    # Tight layout to optimize space usage
    plt.tight_layout()

    # Make sure the output directory exists
    os.makedirs('./output', exist_ok=True)
    
    # Save the figure with high resolution
    plt.savefig('./output/reuse_potential_bar.pdf', bbox_inches='tight')
    plt.savefig('./output/reuse_potential_bar.png', bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    # Define trace types to analyze
    trace_types = ["conversation", "toolagent", "swe_agent"]
    
    # Plot the bar chart of maximum reuse potential
    plot_reuse_potential_bar(trace_types)
