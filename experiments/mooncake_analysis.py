#%%
import json
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import sys

# --- Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze KVCache reuse patterns in trace files')
    parser.add_argument('--trace-file', help='Path to a single trace file')
    parser.add_argument('--trace-files', nargs='+', help='Paths to multiple trace files for comparative analysis')
    parser.add_argument('--block-size', type=int, default=512, help='Block size in tokens')
    parser.add_argument('--output-dir', default='reports', help='Directory to save the report and plots')
    parser.add_argument('--report-name', help='Name for the report file (default: derived from trace file name)')
    parser.add_argument('--no-plots', action='store_true', help='Disable plot generation')
    args = parser.parse_args()
    
    # Ensure at least one trace file is provided
    if not args.trace_file and not args.trace_files:
        parser.error("Either --trace-file or --trace-files must be provided")
    
    # If both are provided, prioritize --trace-files for comparative analysis
    if args.trace_file and not args.trace_files:
        args.trace_files = [args.trace_file]
    
    return args


def load_trace_data(trace_file):
    """Load and parse trace data from a file."""
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
        return None

    if not data:
        print(f"No valid data loaded from {trace_file}.")
        return None
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)  # Reset index after sorting
    
    # Add trace file name as a column for identification in comparative analysis
    df['trace_name'] = os.path.splitext(os.path.basename(trace_file))[0]
    
    return df


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


def analyze_single_trace(df, args, trace_name, run_dir, plots_dir):
    """Analyze a single trace file and generate a report."""
    # Store the original hash IDs before replacing them
    df['original_hash_ids'] = df['hash_ids']

    # Replace hash_ids with running hashes
    df['hash_ids'] = df.apply(lambda row: create_running_hashes(row['original_hash_ids']), axis=1)

    # Verify the transformation
    if len(df) > 0:
        print("Created running hashes for {0} requests".format(len(df)))
    print()

    # --- Basic Statistics ---
    print("=== Basic Statistics ===\n")
    num_requests = len(df)
    total_duration_ms = df['timestamp'].max() - df['timestamp'].min() if num_requests > 1 else 0
    # Avoid division by zero if duration is 0
    avg_request_rate_rps = (num_requests / (total_duration_ms / 1000.0)) if total_duration_ms > 0 else float('inf')

    print("Total requests: {0}".format(num_requests))
    print("Trace duration (s): {0:.2f}".format(total_duration_ms / 1000.0))
    print("Average request rate (req/sec): {0:.2f}".format(avg_request_rate_rps))
    print()

    # --- Input/Output Length Analysis ---
    print("=== Token Length Statistics ===\n")
    df['total_tokens'] = df['input_length'] + df['output_length']
    df['num_hash_blocks'] = df['hash_ids'].apply(len)
    # Approx input length based on blocks, for sanity check
    df['approx_input_from_blocks'] = df['num_hash_blocks'] * args.block_size

    print("Input Length:")
    print(df['input_length'].describe())
    print("\nOutput Length:")
    print(df['output_length'].describe())
    print("\nTotal Tokens (Input + Output):")
    print(df['total_tokens'].describe())
    print()
    
    # Generate plots if enabled
    if not args.no_plots:
        # Plot CDF of input length
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(df['input_length'])
        plt.xlabel('Input Length (tokens)')
        plt.ylabel('CDF')
        plt.title('CDF of Input Length')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "input_length_cdf.png"))
        plt.close()

        # Plot CDF of output length
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(df['output_length'])
        plt.xlabel('Output Length (tokens)')
        plt.ylabel('CDF')
        plt.title('CDF of Output Length')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "output_length_cdf.png"))
        plt.close()

    # --- Arrival Pattern Analysis ---
    print("=== Arrival Pattern Analysis ===\n")
    # Calculate Inter-Arrival Time (IAT) in milliseconds
    df['iat_ms'] = df['timestamp'].diff()
    # First request has no IAT
    df.loc[0, 'iat_ms'] = 0 # Or np.nan if preferred, describe handles nan

    # Exclude the first request's 0 IAT for meaningful stats if needed
    iat_stats = df['iat_ms'][1:].describe() if num_requests > 1 else df['iat_ms'].describe()
    print("Inter-Arrival Time (ms):")
    print(iat_stats)
    
    # Check for bursts (low IATs)
    if num_requests > 1:
        print("\nPercentage of requests arriving within:")
        print("  10ms: {0:.2f}%".format((df['iat_ms'][1:] <= 10).mean() * 100))
        print("  100ms: {0:.2f}%".format((df['iat_ms'][1:] <= 100).mean() * 100))
        print("  1000ms: {0:.2f}%".format((df['iat_ms'][1:] <= 1000).mean() * 100))
    print()
    
    if not args.no_plots and num_requests > 1:
        # Plot CDF of inter-arrival times
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(df['iat_ms'][1:])
        plt.xlabel('Inter-Arrival Time (ms)')
        plt.ylabel('CDF')
        plt.title('CDF of Request Inter-Arrival Times')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "iat_cdf.png"))
        plt.close()

    # --- KVCache Reuse Analysis ---
    print("=== KVCache Reuse Analysis ===\n")
    
    # 1. Calculate prefix match using running hashes and a hash map for efficient lookups
    print("Analyzing prefix matches...")
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
    df['prefix_match_tokens'] = df['prefix_match_blocks'] * args.block_size

    # Statistics on prefix matching
    requests_with_match = df[df['prefix_match_blocks'] > 0]
    num_matches = len(requests_with_match)
    match_percentage = (num_matches / num_requests) * 100 if num_requests > 0 else 0

    print("Requests with any prefix match: {0} ({1:.2f}%)".format(num_matches, match_percentage))

    if num_matches > 0:
        print("\nMatched Prefix Length (Blocks):")
        print(requests_with_match['prefix_match_blocks'].describe())
        print("\nMatched Prefix Length (Approx. Tokens):")
        print(requests_with_match['prefix_match_tokens'].describe())
    else:
        print("\nNo prefix matches found in this sample.")

    # Calculate overall potential reuse ratio (matched tokens / total input tokens)
    total_input_tokens = df['input_length'].sum()
    total_matched_tokens = df['prefix_match_tokens'].sum()
    reuse_ratio = (total_matched_tokens / total_input_tokens) * 100 if total_input_tokens > 0 else 0
    print("\nOverall Potential Reuse Ratio: {0:.2f}%".format(reuse_ratio))
    print()
    
    # Return key metrics for comparative analysis
    metrics = {
        'trace_name': trace_name,
        'num_requests': num_requests,
        'avg_request_rate': avg_request_rate_rps,
        'avg_input_length': df['input_length'].mean(),
        'avg_output_length': df['output_length'].mean(),
        'match_percentage': match_percentage,
        'avg_match_blocks': requests_with_match['prefix_match_blocks'].mean() if num_matches > 0 else 0,
        'reuse_ratio': reuse_ratio
    }
    
    if not args.no_plots and num_matches > 0:
        # Plot CDF of prefix match lengths
        plt.figure(figsize=(10, 6))
        sns.ecdfplot(requests_with_match['prefix_match_blocks'])
        plt.xlabel('Prefix Match Length (Blocks)')
        plt.ylabel('CDF')
        plt.title('CDF of Prefix Match Lengths')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "prefix_match_cdf.png"))
        plt.close()
        
        # Plot scatter of prefix match length vs input length
        plt.figure(figsize=(10, 6))
        plt.scatter(df['input_length'], df['prefix_match_tokens'], alpha=0.5)
        plt.xlabel('Input Length (Tokens)')
        plt.ylabel('Prefix Match Length (Tokens)')
        plt.title('Input Length vs Prefix Match Length')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "prefix_match_scatter.png"))
        plt.close()
        
        # Plot CDF of prefix match length as a fraction of input length
        plt.figure(figsize=(10, 6))
        match_fraction = df['prefix_match_tokens'] / df['input_length']
        match_fraction = match_fraction.fillna(0)  # Replace NaN with 0
        sns.ecdfplot(match_fraction)
        plt.xlabel('Prefix Match Length (Fraction of Input Length)')
        plt.ylabel('CDF')
        plt.title('CDF of Prefix Match Length as a Fraction of Input Length')
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "prefix_match_fraction_cdf.png"))
        plt.close()
        
        # Plot CDF of prefix match length and input length together
        plt.figure(figsize=(10, 6))
        # Filter out the outliers
        prefix_match_tokens_95 = df['prefix_match_tokens'].quantile(0.95)
        input_length_95 = df['input_length'].quantile(0.95)
        sns.ecdfplot(df[df['prefix_match_tokens'] <= prefix_match_tokens_95]['prefix_match_tokens'], label='Prefix Match Length (Tokens)')
        sns.ecdfplot(df[df['input_length'] <= input_length_95]['input_length'], label='Input Length (Tokens)')
        plt.xlabel('Tokens')
        plt.ylabel('CDF')
        plt.title('CDF of Input Length vs Prefix Match Length')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "input_vs_match_cdf.png"))
        plt.close()
    
    # 2. Analyze frequency of individual hash blocks (identifying hot blocks)
    print("Analyzing block frequency...")
    all_hash_ids = [hash_id for sublist in df['original_hash_ids'] for hash_id in sublist]
    if not all_hash_ids:
        print("No hash blocks found in the trace.")
        return metrics, df
    
    hash_id_counts = Counter(all_hash_ids)
    total_blocks = len(all_hash_ids)
    unique_blocks = len(hash_id_counts)
    print("Total blocks: {0}".format(total_blocks))
    print("Unique blocks: {0}".format(unique_blocks))
    
    # Sort by frequency
    sorted_blocks = hash_id_counts.most_common()
    top_block_count = sorted_blocks[0][1] if sorted_blocks else 0
    top_block_percentage = (top_block_count / total_blocks) * 100 if total_blocks > 0 else 0
    print("Most frequent block: {0} occurrences ({1:.2f}% of all blocks)".format(top_block_count, top_block_percentage))
    
    # Calculate concentration metrics
    if sorted_blocks:
        # Initialize these variables with default values
        top_10_percent = 0
        percentage_from_top = 0
        
        # Calculate the number of blocks that make up 10% of total blocks
        cumulative_count = 0
        for i, (block, count) in enumerate(sorted_blocks):
            cumulative_count += count
            if cumulative_count >= total_blocks * 0.1:
                top_10_percent = i + 1
                percentage_from_top = (top_10_percent / unique_blocks) * 100
                break
        
        print("Top {0} blocks ({1:.2f}% of unique blocks) account for 10% of all blocks".format(
            top_10_percent, percentage_from_top))
    
    # Calculate block concentration (percentage of blocks that account for 80% of accesses)
    if sorted_blocks:
        cumulative_count = 0
        blocks_for_80_percent = 0
        for i, (block, count) in enumerate(sorted_blocks):
            cumulative_count += count
            blocks_for_80_percent = i + 1
            if cumulative_count >= total_blocks * 0.8:
                break
        
        block_concentration = (blocks_for_80_percent / unique_blocks) * 100 if unique_blocks > 0 else 0
        print("Block concentration: {0:.2f}% of unique blocks account for 80% of accesses".format(block_concentration))
        metrics['block_concentration'] = block_concentration
    else:
        metrics['block_concentration'] = 0
    
    # 3. Analyze the temporal pattern of reuse
    print("Analyzing temporal reuse patterns...")
    # Create a dictionary to track when each hash_id was last seen
    hash_last_seen = {}
    # Dictionary to store time gaps for each hash_id
    hash_time_gaps = {}

    # Process each request in timestamp order
    df_sorted = df.sort_values(by='timestamp')
    for idx, row in df_sorted.iterrows():
        current_time = row['timestamp']

        # Check each unique hash_id in this request
        for hash_id in row['hash_ids']:
            if hash_id in hash_last_seen:
                # Calculate time gap since last occurrence
                time_gap = current_time - hash_last_seen[hash_id]

                # Store the time gap
                if hash_id not in hash_time_gaps:
                    hash_time_gaps[hash_id] = []
                hash_time_gaps[hash_id].append(time_gap)
            
            # Update the last seen timestamp for this hash_id
            hash_last_seen[hash_id] = current_time

    # Flatten the time gaps for analysis
    all_time_gaps = [gap for gaps in hash_time_gaps.values() for gap in gaps]

    print("Temporal Pattern of Block Reuse:")
    if all_time_gaps:
        # Convert to numpy array for statistics
        all_time_gaps_np = np.array(all_time_gaps)
        
        # Basic statistics
        print("Total block reuses analyzed: {0}".format(len(all_time_gaps)))
        print("Unique blocks that were reused: {0}".format(len(hash_time_gaps)))
        
        # Time gap statistics (in milliseconds)
        print("\nTime gap between reuses (ms):")
        print("  Min: {0:.2f}".format(np.min(all_time_gaps_np)))
        print("  25th percentile: {0:.2f}".format(np.percentile(all_time_gaps_np, 25)))
        print("  Median: {0:.2f}".format(np.median(all_time_gaps_np)))
        print("  Mean: {0:.2f}".format(np.mean(all_time_gaps_np)))
        print("  75th percentile: {0:.2f}".format(np.percentile(all_time_gaps_np, 75)))
        print("  95th percentile: {0:.2f}".format(np.percentile(all_time_gaps_np, 95)))
        print("  Max: {0:.2f}".format(np.max(all_time_gaps_np)))
        
        # Add time gap metrics
        metrics['median_time_gap'] = np.median(all_time_gaps_np)
        metrics['mean_time_gap'] = np.mean(all_time_gaps_np)
        
        # Analyze short-term vs long-term reuse
        short_term_threshold = 1000  # 1 second in ms
        medium_term_threshold = 10000  # 10 seconds in ms
        
        short_term_reuses = sum(gap < short_term_threshold for gap in all_time_gaps)
        medium_term_reuses = sum(short_term_threshold <= gap < medium_term_threshold for gap in all_time_gaps)
        long_term_reuses = sum(gap >= medium_term_threshold for gap in all_time_gaps)
        
        print("\nReuse Time Categories:")
        print("  Short-term (<1s): {0} reuses ({1:.2f}%)".format(short_term_reuses, short_term_reuses/len(all_time_gaps)*100))
        print("  Medium-term (1-10s): {0} reuses ({1:.2f}%)".format(medium_term_reuses, medium_term_reuses/len(all_time_gaps)*100))
        print("  Long-term (>10s): {0} reuses ({1:.2f}%)".format(long_term_reuses, long_term_reuses/len(all_time_gaps)*100))
        
        # Add reuse time categories to metrics
        metrics['short_term_reuse_pct'] = short_term_reuses/len(all_time_gaps)*100 if all_time_gaps else 0
        metrics['medium_term_reuse_pct'] = medium_term_reuses/len(all_time_gaps)*100 if all_time_gaps else 0
        metrics['long_term_reuse_pct'] = long_term_reuses/len(all_time_gaps)*100 if all_time_gaps else 0
    else:
        print("No block reuses found in this dataset.")
    print()
    
    return metrics, df


def generate_comparative_plots(metrics_list, all_dfs, plots_dir):
    """Generate comparative plots for multiple trace files."""
    # Convert metrics list to DataFrame for easier plotting
    metrics_df = pd.DataFrame(metrics_list)
    
    # Create a bar chart for key metrics
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['reuse_ratio', 'match_percentage', 'block_concentration']
    metrics_df_plot = metrics_df[['trace_name'] + metrics_to_plot].melt(
        id_vars=['trace_name'], 
        value_vars=metrics_to_plot,
        var_name='Metric', 
        value_name='Value'
    )
    
    # Create a grouped bar chart
    sns.barplot(x='Metric', y='Value', hue='trace_name', data=metrics_df_plot)
    plt.title('Comparison of Key Metrics Across Traces')
    plt.ylabel('Percentage (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "key_metrics_comparison.png"))
    plt.close()
    
    # Create a bar chart for reuse time categories
    plt.figure(figsize=(12, 8))
    reuse_metrics = ['short_term_reuse_pct', 'medium_term_reuse_pct', 'long_term_reuse_pct']
    reuse_df_plot = metrics_df[['trace_name'] + reuse_metrics].melt(
        id_vars=['trace_name'], 
        value_vars=reuse_metrics,
        var_name='Reuse Category', 
        value_name='Percentage'
    )
    
    # Map category names to more readable labels
    category_map = {
        'short_term_reuse_pct': 'Short-term (<1s)',
        'medium_term_reuse_pct': 'Medium-term (1-10s)',
        'long_term_reuse_pct': 'Long-term (>10s)'
    }
    reuse_df_plot['Reuse Category'] = reuse_df_plot['Reuse Category'].map(category_map)
    
    # Create a grouped bar chart
    sns.barplot(x='Reuse Category', y='Percentage', hue='trace_name', data=reuse_df_plot)
    plt.title('Comparison of Reuse Time Categories Across Traces')
    plt.ylabel('Percentage (%)')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "reuse_time_comparison.png"))
    plt.close()
    
    # Create a bar chart for input/output lengths
    plt.figure(figsize=(12, 8))
    length_metrics = ['avg_input_length', 'avg_output_length']
    length_df_plot = metrics_df[['trace_name'] + length_metrics].melt(
        id_vars=['trace_name'], 
        value_vars=length_metrics,
        var_name='Length Type', 
        value_name='Average Tokens'
    )
    
    # Map length types to more readable labels
    length_map = {
        'avg_input_length': 'Average Input Length',
        'avg_output_length': 'Average Output Length'
    }
    length_df_plot['Length Type'] = length_df_plot['Length Type'].map(length_map)
    
    # Create a grouped bar chart
    sns.barplot(x='Length Type', y='Average Tokens', hue='trace_name', data=length_df_plot)
    plt.title('Comparison of Average Token Lengths Across Traces')
    plt.ylabel('Tokens')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "token_length_comparison.png"))
    plt.close()
    
    # Additional comparative plots
    
    # 1. Comparative CDF of prefix match fraction
    plt.figure(figsize=(12, 8))
    for i, df in enumerate(all_dfs):
        trace_name = df['trace_name'].iloc[0]
        match_fraction = df['prefix_match_tokens'] / df['input_length']
        match_fraction = match_fraction.fillna(0)  # Replace NaN with 0
        sns.ecdfplot(match_fraction, label=trace_name)
    plt.xlabel('Prefix Match Length (Fraction of Input Length)')
    plt.ylabel('CDF')
    plt.title('CDF of Prefix Match Length as a Fraction of Input Length - Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "prefix_match_fraction_cdf_comparison.png"))
    plt.close()
    
    # 2. Comparative CDF of time gaps
    plt.figure(figsize=(12, 8))
    for i, df in enumerate(all_dfs):
        if 'time_gaps' in df.columns:
            trace_name = df['trace_name'].iloc[0]
            # Collect all time gaps, filtering out None, empty lists, and non-list values
            all_time_gaps = []
            for gaps in df['time_gaps'].dropna():
                if isinstance(gaps, list) and gaps:
                    all_time_gaps.extend(gaps)
            
            if all_time_gaps:
                # Filter out extreme outliers
                time_gaps_99 = np.percentile(all_time_gaps, 99)
                filtered_time_gaps = [gap for gap in all_time_gaps if gap <= time_gaps_99]
                sns.ecdfplot(filtered_time_gaps, label=trace_name)
    
    if plt.gca().get_lines():  # Check if any lines were plotted
        plt.xlabel('Time Gap (ms)')
        plt.ylabel('CDF')
        plt.title('CDF of Time Gaps Between Block Reuses - Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "time_gap_cdf_comparison.png"))
    plt.close()
    
    # 3. Comparative scatter plot of prefix match vs input length
    plt.figure(figsize=(12, 8))
    for i, df in enumerate(all_dfs):
        trace_name = df['trace_name'].iloc[0]
        plt.scatter(df['input_length'], df['prefix_match_tokens'], alpha=0.3, label=trace_name)
    plt.xlabel('Input Length (Tokens)')
    plt.ylabel('Prefix Match Length (Tokens)')
    plt.title('Input Length vs Prefix Match Length - Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, "prefix_match_scatter_comparison.png"))
    plt.close()
    
    # 4. Comparative block popularity distribution
    plt.figure(figsize=(12, 8))
    for i, metrics in enumerate(metrics_list):
        if 'sorted_rates' in metrics and metrics['sorted_rates']:
            trace_name = metrics['trace_name']
            sorted_rates = metrics['sorted_rates']
            plt.plot(range(1, len(sorted_rates) + 1), sorted_rates, label=trace_name)
    
    if plt.gca().get_lines():  # Check if any lines were plotted
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Block Rank (log scale)')
        plt.ylabel('Frequency (log scale)')
        plt.title('Block Popularity Distribution - Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, "block_popularity_comparison.png"))
    plt.close()


def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if we're doing comparative analysis
    if len(args.trace_files) > 1:
        # Create a directory for comparative analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparative_dir = os.path.join(args.output_dir, "comparative_analysis_{0}".format(timestamp))
        os.makedirs(comparative_dir, exist_ok=True)
        
        # Create a plots directory for comparative plots
        plots_dir = os.path.join(comparative_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up the report file path
        report_file = os.path.join(comparative_dir, "comparative_report.txt")
        
        # Redirect stdout to the report file
        original_stdout = sys.stdout
        with open(report_file, 'w') as f:
            sys.stdout = f
            
            print("=== Comparative KVCache Reuse Analysis Report ===\n")
            print("Traces Analyzed:")
            for i, trace_file in enumerate(args.trace_files):
                print("  {0}. {1}".format(i+1, trace_file))
            print("Block Size: {0}".format(args.block_size))
            print("Generated: {0}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # Load and analyze each trace file
            all_metrics = []
            all_dfs = []
            
            for trace_file in args.trace_files:
                trace_name = os.path.splitext(os.path.basename(trace_file))[0]
                print("\n" + "=" * 80)
                print("Analyzing Trace: {0}".format(trace_file))
                print("=" * 80 + "\n")
                
                # Load trace data
                df = load_trace_data(trace_file)
                if df is None:
                    continue
                
                # Create a directory for this trace's individual analysis
                trace_dir = os.path.join(comparative_dir, trace_name)
                os.makedirs(trace_dir, exist_ok=True)
                
                # Create a plots directory for this trace
                trace_plots_dir = os.path.join(trace_dir, "plots")
                os.makedirs(trace_plots_dir, exist_ok=True)
                
                # Analyze the trace
                metrics, df_analyzed = analyze_single_trace(df, args, trace_name, trace_dir, trace_plots_dir)
                
                # Process block frequency data for comparative plots
                all_hash_ids = [hash_id for sublist in df_analyzed['original_hash_ids'] for hash_id in sublist]
                if all_hash_ids:
                    hash_id_counts = Counter(all_hash_ids)
                    block_hit_rates = {hash_id: count/len(all_hash_ids) for hash_id, count in hash_id_counts.items()}
                    sorted_rates = sorted(block_hit_rates.values(), reverse=True)
                    metrics['sorted_rates'] = sorted_rates
                
                # Process time gaps for comparative plots
                hash_time_gaps = {}
                hash_last_seen = {}
                for idx, row in df_analyzed.sort_values(by='timestamp').iterrows():
                    current_time = row['timestamp']
                    for hash_id in row['original_hash_ids']:
                        if hash_id in hash_last_seen:
                            time_gap = current_time - hash_last_seen[hash_id]
                            if hash_id not in hash_time_gaps:
                                hash_time_gaps[hash_id] = []
                            hash_time_gaps[hash_id].append(time_gap)
                        hash_last_seen[hash_id] = current_time
                
                # Store time gaps for each request
                df_analyzed['time_gaps'] = [[gap for gaps in hash_time_gaps.values() for gap in gaps]] * len(df_analyzed)
                
                all_metrics.append(metrics)
                all_dfs.append(df_analyzed)
            
            # Generate comparative analysis
            if len(all_metrics) > 1:
                print("\n" + "=" * 80)
                print("Comparative Analysis")
                print("=" * 80 + "\n")
                
                # Create a table of key metrics
                metrics_df = pd.DataFrame(all_metrics)
                print("Key Metrics Comparison:")
                print(metrics_df.set_index('trace_name')[["num_requests", "avg_request_rate", "avg_input_length", "avg_output_length", "match_percentage", "reuse_ratio", "block_concentration"]])
                
                # Compare reuse patterns
                print("\nReuse Pattern Comparison:")
                reuse_df = metrics_df.set_index('trace_name')[["short_term_reuse_pct", "medium_term_reuse_pct", "long_term_reuse_pct", "median_time_gap", "mean_time_gap"]]
                reuse_df.columns = ["Short-term (<1s) %", "Medium-term (1-10s) %", "Long-term (>10s) %", "Median Time Gap (ms)", "Mean Time Gap (ms)"]
                print(reuse_df)
                
                # Generate comparative plots
                if not args.no_plots:
                    generate_comparative_plots(all_metrics, all_dfs, plots_dir)
        
        # Restore stdout
        sys.stdout = original_stdout
        print("Comparative analysis complete! Report saved to: {0}".format(report_file))
        print("Comparative plots saved to: {0}".format(plots_dir))
        print("Individual trace analyses saved to subdirectories of: {0}".format(comparative_dir))
    
    else:  # Single trace analysis
        trace_file = args.trace_files[0]
        
        # Get the trace file name without extension and path
        if args.report_name:
            trace_name = args.report_name
        else:
            trace_name = os.path.splitext(os.path.basename(trace_file))[0]
        
        # Create timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a specific directory for this trace and timestamp
        trace_dir = os.path.join(args.output_dir, trace_name)
        run_dir = os.path.join(trace_dir, timestamp)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create a plots directory within the run directory
        plots_dir = os.path.join(run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Set up the report file path
        report_file = os.path.join(run_dir, "report.txt")
        
        # Load trace data
        df = load_trace_data(trace_file)
        if df is None:
            print("Error: Could not load trace file {0}".format(trace_file))
            return
        
        # Redirect stdout to the report file
        original_stdout = sys.stdout
        with open(report_file, 'w') as f:
            sys.stdout = f
            
            print("=== KVCache Reuse Analysis Report ===\n")
            print("Trace File: {0}".format(trace_file))
            print("Block Size: {0}".format(args.block_size))
            print("Generated: {0}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            # Analyze the trace
            analyze_single_trace(df, args, trace_name, run_dir, plots_dir)
        
        # Restore stdout
        sys.stdout = original_stdout
        print("Analysis complete! Report saved to: {0}".format(report_file))
        print("Plots saved to: {0}".format(plots_dir))


if __name__ == "__main__":
    main()
