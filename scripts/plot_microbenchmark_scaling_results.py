import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional

# --- Configuration ---
STATS_FILENAME_PATTERNS = {
    "prefill": "prefill_stats.json",
    "decode": "decode_stats.json",
}
# Regex to parse directory names like 'prefill_sglang_tp1_pp1' or 'decode_vajra_tp4_pp2' etc.
# Allows for engine names with hyphens, dots or numbers if needed
DIR_PATTERN = re.compile(r"^(prefill|decode)_([a-zA-Z0-9._-]+)_tp(\d+)_pp(\d+)$") # Allow '.', '_', '-'

# --- Fixed Engine Order ---
FIXED_ENGINE_ORDER = ["vajra", "vllm", "sglang"]

# --- Helper Functions ---

def parse_benchmark_key(key_str: str, profile_type: str) -> tuple:
    """Converts benchmark keys ('512', '512_8') into sortable tuples."""
    if profile_type == "prefill":
        try:
            return (int(key_str),) # Sort by length
        except ValueError:
            return (math.inf,) # Put unparseable keys last
    elif profile_type == "decode":
        try:
            parts = key_str.split('_')
            if len(parts) == 2:
                return (int(parts[0]), int(parts[1])) # Sort by context, then batch
            else:
                 return (math.inf, math.inf)
        except (ValueError, IndexError):
            return (math.inf, math.inf)
    else:
        return (key_str,) # Default alphabetical sort if type unknown

def find_and_parse_results(base_dir: str) -> dict:
    """
    Traverses the base directory, finds result directories, parses stats files.

    Returns:
        A nested dictionary: results[tp_pp_key][profile_type][engine][benchmark_key] = median_latency
        Example: results['tp1_pp1']['prefill']['sglang']['512'] = 0.064
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) # tp_pp -> profile -> engine -> bench_key -> latency

    print(f"Scanning directory: {base_dir}")
    for item_name in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item_name)
        if not os.path.isdir(item_path):
            continue

        match = DIR_PATTERN.match(item_name)
        if not match:
             print(f"  Skipping directory (does not match pattern): {item_name}")
             continue

        profile_type, engine_raw, tp_str, pp_str = match.groups()
        tp = int(tp_str)
        pp = int(pp_str)

        # Clean up potential model names included in the engine part (basic heuristic)
        engine_clean = engine_raw.split('_')[0].split('-')[0] # Try to get 'sglang' from 'sglang_llama...'
        if engine_clean not in FIXED_ENGINE_ORDER:
             # Fallback if simple split doesn't work, check if known engine is substring
             found = False
             for known_engine in FIXED_ENGINE_ORDER:
                  if known_engine in engine_raw:
                       engine_clean = known_engine
                       found = True
                       break
             if not found:
                  print(f"  Warning: Could not reliably determine engine type from '{engine_raw}' in {item_name}. Using '{engine_clean}'.")


        tp_pp_key = f"tp{tp}_pp{pp}"
        stats_filename = STATS_FILENAME_PATTERNS.get(profile_type)

        if not stats_filename:
            # This case should technically not happen if DIR_PATTERN matched
            print(f"  Skipping directory (unknown profile type): {item_name}")
            continue

        stats_filepath = os.path.join(item_path, stats_filename)

        # Look inside potential model subfolder if stats file not found directly
        if not os.path.isfile(stats_filepath):
            found_in_subdir = False
            try:
                for sub_item in os.listdir(item_path):
                    sub_item_path = os.path.join(item_path, sub_item)
                    if os.path.isdir(sub_item_path):
                        potential_stats_path = os.path.join(sub_item_path, stats_filename)
                        if os.path.isfile(potential_stats_path):
                            stats_filepath = potential_stats_path
                            # print(f"  Found stats file in subdirectory: {stats_filepath}")
                            found_in_subdir = True
                            break
            except OSError: # Handle cases where listing might fail
                 pass
            if not found_in_subdir:
                 print(f"  Warning: Stats file not found for {item_name}. Searched: {os.path.join(item_path, stats_filename)} and subdirs.")
                 continue


        try:
            with open(stats_filepath, 'r') as f:
                stats_data = json.load(f)

            if not isinstance(stats_data, dict):
                 print(f"  Warning: Invalid stats file format (not a dict) for {item_name}: {stats_filepath}")
                 continue

            for benchmark_key, benchmark_stats in stats_data.items():
                if isinstance(benchmark_stats, dict) and "median" in benchmark_stats:
                    median_latency = benchmark_stats["median"]
                    results[tp_pp_key][profile_type][engine_clean][benchmark_key] = median_latency
                else:
                    print(f"  Warning: Missing 'median' value for key '{benchmark_key}' in {stats_filepath}")

            # print(f"  Successfully parsed: {item_name} (Engine: {engine_clean})")

        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {stats_filepath}")
        except IOError as e:
            print(f"  Error: Could not read file {stats_filepath}: {e}")
        except Exception as e:
             print(f"  Error: Unexpected error processing {stats_filepath}: {e}")

    if not results:
         print("No valid result directories found.")

    return results

def restructure_for_scaling(parsed_data: dict) -> Tuple[dict, dict]:
    """
    Restructures parsed data for easy scaling plot generation.

    Returns:
        Tuple: (tp_scaling_data, pp_scaling_data)
        tp_scaling_data[profile][benchmark_key][engine][tp] = latency (where pp=1)
        pp_scaling_data[profile][benchmark_key][engine][pp] = latency (where tp=1)
    """
    tp_scaling_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pp_scaling_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for tp_pp_key, profile_data in parsed_data.items():
        match = re.match(r"tp(\d+)_pp(\d+)", tp_pp_key)
        if not match: continue
        tp, pp = map(int, match.groups())

        for profile_type, engine_data in profile_data.items():
            for engine, benchmark_data in engine_data.items():
                for benchmark_key, latency in benchmark_data.items():
                    if pp == 1:
                        tp_scaling_data[profile_type][benchmark_key][engine][tp] = latency
                    if tp == 1:
                        pp_scaling_data[profile_type][benchmark_key][engine][pp] = latency

    return tp_scaling_data, pp_scaling_data


def plot_grouped_comparison(tp_pp_key: str, profile_type: str, engine_data: Dict[str, Dict[str, float]], output_dir: str):
    """
    Generates and saves a single comparison bar chart for a specific TP/PP config and profile type.
    Ensures fixed engine order and adds relative multipliers.
    """
    # (Code from previous answer - unchanged)
    if not engine_data:
        # print(f"No data to plot for grouped comparison {tp_pp_key} - {profile_type}")
        return

    present_engines = [eng for eng in FIXED_ENGINE_ORDER if eng in engine_data]
    if not present_engines: return
    reference_engine = present_engines[0]
    all_benchmark_keys_unsorted = set().union(*(engine_data[eng].keys() for eng in present_engines))
    if not all_benchmark_keys_unsorted: return
    all_benchmark_keys = sorted(list(all_benchmark_keys_unsorted), key=lambda k: parse_benchmark_key(k, profile_type))
    reference_latencies = {key: engine_data[reference_engine].get(key, np.nan) for key in all_benchmark_keys}

    n_engines = len(present_engines)
    n_benchmarks = len(all_benchmark_keys)
    bar_width_total_group = 0.8
    bar_width_individual = bar_width_total_group / n_engines
    index = np.arange(n_benchmarks)

    fig, ax = plt.subplots(figsize=(max(10, n_benchmarks * n_engines * 0.35), 7))

    for i, engine in enumerate(present_engines):
        latencies = [engine_data[engine].get(key, np.nan) for key in all_benchmark_keys]
        bar_positions = index - (bar_width_total_group / 2) + (i * bar_width_individual) + (bar_width_individual / 2)
        bars = ax.bar(bar_positions, latencies, bar_width_individual, label=engine)

        for bar_idx, bar in enumerate(bars):
            key = all_benchmark_keys[bar_idx]
            current_latency = bar.get_height()
            ref_latency = reference_latencies.get(key, np.nan)
            multiplier_text = ""
            if not np.isnan(current_latency):
                if engine == reference_engine: multiplier_text = "1.0x"
                elif not np.isnan(ref_latency) and ref_latency > 1e-9:
                    multiplier = current_latency / ref_latency
                    multiplier_text = f"{multiplier:.1f}x"
                else: multiplier_text = "N/A"
            if multiplier_text:
                ax.text(bar.get_x() + bar.get_width() / 2., current_latency, multiplier_text,
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Median Latency (seconds)')
    ax.set_xticks(index)
    ax.set_xticklabels(all_benchmark_keys, rotation=30, ha='right')
    ax.set_title(f'{profile_type.capitalize()} Latency Comparison ({reference_engine}=1.0x) - {tp_pp_key}')
    ax.legend(title="Engine", loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.1)

    all_positive_latencies = [l for key in all_benchmark_keys for eng in present_engines if (l := engine_data[eng].get(key, 0)) > 0]
    if all_positive_latencies and (max(all_positive_latencies) / min(all_positive_latencies) > 50):
        ax.set_yscale('log')
        ax.set_ylabel('Median Latency (seconds, log scale)')
        # print(f"  Using log scale for y-axis in grouped {profile_type} {tp_pp_key} plot.")
        current_ylim_log = ax.get_ylim()
        ax.set_ylim(current_ylim_log[0], current_ylim_log[1] * 1.5)

    if profile_type == 'prefill': ax.set_xlabel('Prefill Length (tokens)')
    elif profile_type == 'decode': ax.set_xlabel('Context Length _ Batch Size')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_filename = f"{profile_type}_{tp_pp_key}_comparison.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_filepath, dpi=150)
        # print(f"  Saved grouped comparison plot: {plot_filepath}")
    except Exception as e:
        print(f"  Error saving grouped plot {plot_filepath}: {e}")
    plt.close(fig)


def plot_scaling(
    scaling_type: str, # "TP" or "PP"
    profile_type: str,
    benchmark_key: str,
    engine_scaling_data: Dict[str, Dict[int, float]], # { engine: {dimension_value: latency} }
    output_dir: str):
    """
    Generates and saves a line plot showing TP or PP scaling for a specific benchmark key.
    """
    if not engine_scaling_data:
        # print(f"No scaling data to plot for {scaling_type} scaling - {profile_type} - {benchmark_key}")
        return

    # --- Data Preparation ---
    present_engines = [eng for eng in FIXED_ENGINE_ORDER if eng in engine_scaling_data]
    if not present_engines:
        # print(f"No engines from {FIXED_ENGINE_ORDER} found for scaling plot: {scaling_type} - {profile_type} - {benchmark_key}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_dim_values = set() # Keep track of all TP/PP values plotted for setting xticks

    # --- Plotting ---
    for engine in present_engines:
        scaling_points = engine_scaling_data.get(engine, {})
        if not scaling_points or len(scaling_points) < 1: # Need at least one point to plot
            continue

        # Sort points by scaling dimension (TP or PP value)
        sorted_points = sorted(scaling_points.items())
        dim_values = [p[0] for p in sorted_points]
        latencies = [p[1] for p in sorted_points]
        all_dim_values.update(dim_values)

        ax.plot(dim_values, latencies, marker='o', linestyle='-', label=engine) # Added linestyle

    # --- Formatting ---
    if not all_dim_values: # No lines were plotted
        plt.close(fig)
        return

    x_label = f"{'Tensor' if scaling_type == 'TP' else 'Pipeline'} Parallelism ({scaling_type})"
    ax.set_xlabel(x_label)
    ax.set_ylabel('Median Latency (seconds)')
    ax.set_title(f'{profile_type.capitalize()} Latency vs {scaling_type} Scaling (Benchmark: {benchmark_key})')
    ax.legend(title="Engine")
    ax.grid(axis='both', linestyle='--', alpha=0.7) # Grid on both axes

    # Set X ticks explicitly to the TP/PP values tested
    sorted_dim_values = sorted(list(all_dim_values))
    ax.set_xticks(sorted_dim_values)
    ax.set_xticklabels(sorted_dim_values) # Use values directly as labels

    # Optional: Log scale for Y-axis (latency)
    all_positive_latencies = [l for eng_data in engine_scaling_data.values() for l in eng_data.values() if l > 0]
    if all_positive_latencies:
         min_lat, max_lat = min(all_positive_latencies), max(all_positive_latencies)
         if max_lat / min_lat > 30: # Threshold for log scale on Y
             ax.set_yscale('log')
             ax.set_ylabel('Median Latency (seconds, log scale)')
             print(f"  Using log scale for y-axis in scaling plot: {scaling_type} - {profile_type} - {benchmark_key}")

    # Optional: Log scale for X-axis (TP/PP) if values are powers of 2 like
    is_power_of_2 = all(v > 0 and math.log2(v).is_integer() for v in sorted_dim_values if v > 0)
    # if len(sorted_dim_values) > 2 and is_power_of_2:
    #     ax.set_xscale('log', base=2)
    #     ax.set_xlabel(f"{x_label} (log scale base 2)")
    #     # Regenerate ticks/labels for log scale if needed, but defaults might be okay
    #     ax.set_xticks(sorted_dim_values)
    #     ax.set_xticklabels(sorted_dim_values)
    #     print(f"  Using log scale for x-axis in scaling plot: {scaling_type} - {profile_type} - {benchmark_key}")


    plt.tight_layout()

    # --- Saving ---
    plot_filename = f"{profile_type}_{benchmark_key}_{scaling_type}_scaling.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    try:
        plt.savefig(plot_filepath, dpi=150)
        # print(f"  Saved scaling plot: {plot_filepath}")
    except Exception as e:
        print(f"  Error saving scaling plot {plot_filepath}: {e}")

    plt.close(fig)

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Generate comparison and scaling plots from Veeksha benchmark results.")
    parser.add_argument("results_dir", type=str,
                        help="Path to the base directory containing the benchmark result folders (e.g., 'engine_microbenchmark_logs').")
    parser.add_argument("-o", "--output-dir", type=str, default="benchmark_plots",
                        help="Directory where the comparison plots will be saved.")
    parser.add_argument("--skip-grouped", action="store_true", help="Skip generating grouped bar charts.")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip generating scaling line plots.")

    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        return

    # Create separate subdirectories for plot types
    grouped_output_dir = os.path.join(args.output_dir, "grouped_comparisons")
    scaling_output_dir = os.path.join(args.output_dir, "scaling_plots")
    if not args.skip_grouped: os.makedirs(grouped_output_dir, exist_ok=True)
    if not args.skip_scaling: os.makedirs(scaling_output_dir, exist_ok=True)
    print(f"Saving grouped plots to: {grouped_output_dir}")
    print(f"Saving scaling plots to: {scaling_output_dir}")

    # 1. Find and parse all results
    parsed_data = find_and_parse_results(args.results_dir)

    if not parsed_data:
        print("No results found to plot.")
        return

    # 2. Generate Grouped Comparison Plots (Optional)
    if not args.skip_grouped:
        print("\n--- Generating Grouped Comparison Plots ---")
        sorted_tp_pp_keys = sorted(parsed_data.keys(), key=lambda k: tuple(map(int, re.findall(r'\d+', k))))
        plot_count = 0
        for tp_pp_key in sorted_tp_pp_keys:
            profile_data = parsed_data[tp_pp_key]
            # print(f"\nProcessing grouped plots for: {tp_pp_key}")
            if "prefill" in profile_data:
                plot_grouped_comparison(tp_pp_key, "prefill", profile_data["prefill"], grouped_output_dir)
                plot_count += 1
            if "decode" in profile_data:
                plot_grouped_comparison(tp_pp_key, "decode", profile_data["decode"], grouped_output_dir)
                plot_count += 1
        print(f"Generated {plot_count} grouped comparison plots.")
    else:
        print("\n--- Skipping Grouped Comparison Plots ---")


    # 3. Restructure Data and Generate Scaling Plots (Optional)
    if not args.skip_scaling:
        print("\n--- Generating Scaling Plots ---")
        tp_scaling_data, pp_scaling_data = restructure_for_scaling(parsed_data)

        # Plot TP Scaling (PP=1)
        plot_count_tp = 0
        print("  Processing TP scaling (PP=1)...")
        for profile_type, bench_data in tp_scaling_data.items():
             # Sort benchmark keys for consistent plot order
             sorted_bench_keys = sorted(bench_data.keys(), key=lambda k: parse_benchmark_key(k, profile_type))
             for benchmark_key in sorted_bench_keys:
                 engine_data = bench_data[benchmark_key]
                 plot_scaling("TP", profile_type, benchmark_key, engine_data, scaling_output_dir)
                 plot_count_tp += 1
        print(f"  Generated {plot_count_tp} TP scaling plots.")

        # Plot PP Scaling (TP=1)
        plot_count_pp = 0
        print("  Processing PP scaling (TP=1)...")
        for profile_type, bench_data in pp_scaling_data.items():
             sorted_bench_keys = sorted(bench_data.keys(), key=lambda k: parse_benchmark_key(k, profile_type))
             for benchmark_key in sorted_bench_keys:
                 engine_data = bench_data[benchmark_key]
                 plot_scaling("PP", profile_type, benchmark_key, engine_data, scaling_output_dir)
                 plot_count_pp += 1
        print(f"  Generated {plot_count_pp} PP scaling plots.")
    else:
        print("\n--- Skipping Scaling Plots ---")


    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()