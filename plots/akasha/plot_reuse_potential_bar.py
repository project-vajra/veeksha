import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from collections import defaultdict

# Import constants from the constants file
from constants import COLORS, OPACITY, FONT, HATCHES, PRETTY_NAMES, SYSTEM_NAME_MAP


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


# Define the simulated reuse rates for different systems
# These values represent what percentage of the maximum potential reuse is actually attained
# by each system (based on your requirements)
SYSTEM_EFFICIENCY = {
    "vllm": 0.65,  # vLLM achieves 65% of potential reuse
    "sglang": 0.75,  # SGLang achieves 75% of potential reuse
    "sglang_wts": 0.85,  # SGLang-WTS achieves 85% of potential reuse
}

# Define pretty names for trace types
TRACE_TYPE_NAMES = {
    "conversation": "Conversation",
    "toolagent": "Tool Agent",
    "swe_agent": "SWE Agent"
}


def plot_reuse_potential_bar(trace_types):
    """Plot a bar chart showing the maximum reuse potential and actual reuse for different systems."""
    # Set up the figure with appropriate size for a single column in SOSP paper
    plt.rcParams.update({'font.size': 10})
    plt.rcParams.update({'font.family': FONT})

    # Create a figure optimized for single column in SOSP paper (typically ~3.33 inches wide)
    fig, ax = plt.subplots(figsize=(3.33, 2.5))  # Match the CDF plot dimensions

    # Create a dictionary to store results for each trace type
    max_potential = {}
    actual_reuse = defaultdict(dict)

    # Load data for each trace type
    for trace_type in trace_types:
        reuse_potential = load_and_process_data(trace_type)
        if reuse_potential > 0:
            max_potential[trace_type] = reuse_potential
            
            # Calculate simulated actual reuse for each system
            for system, efficiency in SYSTEM_EFFICIENCY.items():
                actual_reuse[system][trace_type] = reuse_potential * efficiency

    # Prepare data for plotting
    trace_labels = [TRACE_TYPE_NAMES.get(t, t.capitalize()) for t in max_potential.keys()]
    x_pos = np.arange(len(trace_labels))
    width = 0.2  # Width of each bar
    
    # Calculate positions for grouped bars - no gaps, no overlaps
    positions = {
        'vllm': x_pos - width * 1.5,
        'sglang': x_pos - width * 0.5,
        'sglang_wts': x_pos + width * 0.5,
        'max': x_pos + width * 1.5
    }
    
    # Plot the bars for each system
    system_bars = {}
    for idx, (system, color_idx) in enumerate(zip(['vllm', 'sglang', 'sglang_wts', 'max'], [1, 2, 3, 0])):
        values = [actual_reuse[system][t] if system != 'max' else max_potential[t] for t in max_potential.keys()]
        
        # Use appropriate label for each system
        if system == 'max':
            label = 'Maximum Potential'
        else:
            label = SYSTEM_NAME_MAP.get(system, PRETTY_NAMES.get(system, system.upper()))
            
        system_bars[system] = ax.bar(positions[system], values, width=width, alpha=OPACITY, label=label)
        
        # Color and add hatches to the system bars
        for i, bar in enumerate(system_bars[system]):
            bar.set_color(COLORS[color_idx])
            bar.set_hatch(HATCHES[color_idx])
    
    # Configure the plot
    ax.grid(which='major', axis='y', linestyle='--', alpha=0.7)
    ax.set_xlabel("Trace Type", fontsize=10, fontweight='bold')
    ax.set_ylabel("Reuse Potential (%)", fontsize=10, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(trace_labels)
    
    # Set y-axis limit with some padding
    max_value = max(max_potential.values()) if max_potential else 0
    ax.set_ylim([0, max_value * 1.4])  # Add 10% padding at the top
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add value labels on top of each bar
    for system, bars in system_bars.items():
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f"{height:.1f}%", ha='center', va='bottom', fontsize=6, rotation=90)

    # Add legend with smaller font size
    ax.legend(loc='upper left', fontsize=6)

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
