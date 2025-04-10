import os
import json
import re
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional, Literal
from pathlib import Path
import itertools

# Try importing seaborn for better palettes, but have fallbacks
try:
    import seaborn as sns
    _SEABORN_AVAILABLE = True
except ImportError:
    _SEABORN_AVAILABLE = False
    print("Warning: Seaborn not found. Using default Matplotlib palettes.")


# --- Configuration ---
STATS_FILENAME_MAP = {
    "veeksha_prefill": "prefill_stats.json",
    "veeksha_decode": "decode_stats.json",
    "veeksha_mixed_batching": "decode_stats.json",
}
BENCHMARK_SUBDIR_PREFIX = "benchmark_"
TOP_DIR_PATTERN = re.compile(r"^([a-zA-Z0-9._-]+)_tp(\d+)_pp(\d+)$") # Captures full name part

# --- Plotting Styles ---
if _SEABORN_AVAILABLE:
    VAJRA_BLUES_PALETTE = sns.color_palette("Blues", 8)[2:] # Get more darker blues if many vajra variants
    OTHER_ENGINE_PALETTE = sns.color_palette("tab10")
else:
    VAJRA_BLUES_PALETTE = plt.cm.get_cmap('Blues', 8)(np.linspace(0.3, 1, 6))
    OTHER_ENGINE_PALETTE = plt.cm.get_cmap('tab10').colors

PLOT_HATCHES = ['/', '\\', '|', '-', '+', 'x', 'o', '*']
PLOT_OPACITY = 0.85
PLOT_MARKERS = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
PLOT_LINESTYLES = ['-', '--', '-.', ':']

# Font settings
LABEL_FONTSIZE = 14
LABEL_FONTWEIGHT = "bold"
TITLE_FONTSIZE = 15
TICK_LEGEND_FONTSIZE = 11
MULTIPLIER_FONTSIZE = 9.5

# --- Type Hint for Metric ---
MetricType = Literal["mean", "median"]

# --- Helper Functions ---

def parse_benchmark_key(key_str: str, profile_type: str) -> tuple:
    """Converts benchmark keys ('512', '512_8') into sortable tuples."""
    profile_type_clean = profile_type.split('veeksha_')[-1]
    if profile_type_clean == "prefill":
        try: return (int(key_str),)
        except ValueError: return (math.inf,)
    elif profile_type_clean == "decode":
        try:
            parts = key_str.split('_')
            return (int(parts[0]), int(parts[1])) if len(parts) == 2 else (math.inf, math.inf)
        except (ValueError, IndexError): return (math.inf, math.inf)
    else: return (key_str,)

def find_and_parse_results(base_dir: str, metric: MetricType) -> dict:
    """
    Finds and parses results, using the full engine name part from the directory
    as the key (e.g., 'vajra_with_overlap').
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    base_path = Path(base_dir)
    print(f"Scanning directory: {base_path.resolve()} (using metric: '{metric}')")
    found_engines = set()

    for top_item_path in base_path.iterdir():
        if not top_item_path.is_dir(): continue
        top_dir_name = top_item_path.name
        match = TOP_DIR_PATTERN.match(top_dir_name)
        if not match: continue

        # Use the full matched name part as the engine key
        engine_key, tp_str, pp_str = match.groups()
        tp = int(tp_str); pp = int(pp_str)
        tp_pp_key = f"tp{tp}_pp{pp}"
        found_engines.add(engine_key) # Add the actual key found

        try:
            for bench_subdir_path in top_item_path.iterdir():
                # [ Rest of the file finding logic remains the same ]
                if not bench_subdir_path.is_dir() or not bench_subdir_path.name.startswith(BENCHMARK_SUBDIR_PREFIX): continue
                bench_subdir_name = bench_subdir_path.name
                profile_type_key = bench_subdir_name.split(BENCHMARK_SUBDIR_PREFIX, 1)[-1]
                stats_filename = STATS_FILENAME_MAP.get(profile_type_key)
                if not stats_filename: continue
                stats_filepath = None; found_stats_file = False
                direct_path = bench_subdir_path / stats_filename
                if direct_path.is_file(): stats_filepath = direct_path; found_stats_file = True
                else:
                    try:
                        for potential_output_dir in bench_subdir_path.iterdir():
                            if potential_output_dir.is_dir():
                                deeper_stats_path = potential_output_dir / stats_filename
                                if deeper_stats_path.is_file(): stats_filepath = deeper_stats_path; found_stats_file = True; break
                    except OSError as list_err: print(f"  Warning: Could not list contents of {bench_subdir_path}: {list_err}")
                if not found_stats_file:
                    if profile_type_key in STATS_FILENAME_MAP: print(f"  Warning: Stats file '{stats_filename}' not found in or one level deep within {bench_subdir_path}")
                    continue
                try:
                    with open(stats_filepath, 'r') as f: stats_data = json.load(f)
                    if not isinstance(stats_data, dict): print(f"  Warning: Invalid stats format (not dict): {stats_filepath}"); continue
                    for benchmark_key, benchmark_stats in stats_data.items():
                        if isinstance(benchmark_stats, dict) and metric in benchmark_stats:
                            metric_value = benchmark_stats[metric]
                            if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                                # Use the full engine_key derived from the directory name
                                results[tp_pp_key][profile_type_key][engine_key][benchmark_key] = float(metric_value)
                            elif metric_value is not None: print(f"  Warning: Non-numeric or NaN '{metric}' value ({metric_value}) for key '{benchmark_key}' in {stats_filepath}")
                except json.JSONDecodeError: print(f"  Error: Could not decode JSON from {stats_filepath}")
                except IOError as e: print(f"  Error: Could not read file {stats_filepath}: {e}")
                except Exception as e: print(f"  Error: Unexpected error processing {stats_filepath}: {e}")
        except OSError as e: print(f"  Error listing directory contents for {top_item_path}: {e}")

    if not results:
         print(f"No valid benchmark results found containing metric '{metric}' in '{base_dir}'.")
    else:
         # Print the actual engine keys found in the data structure
         all_found_keys = set(eng for tp_data in results.values() for prof_data in tp_data.values() for eng in prof_data.keys())
         print(f"Found data for engines: {sorted(list(all_found_keys))}")
    return results


def restructure_for_scaling(parsed_data: dict) -> Tuple[dict, dict]:
    """
    Restructures parsed data for easy scaling plot generation.
    [Function content remains the same]
    """
    tp_scaling_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    pp_scaling_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for tp_pp_key, profile_data in parsed_data.items():
        match = re.match(r"tp(\d+)_pp(\d+)", tp_pp_key)
        if not match: continue
        tp, pp = map(int, match.groups())
        for profile_type, engine_data in profile_data.items():
            for engine, benchmark_data in engine_data.items():
                for benchmark_key, metric_value in benchmark_data.items():
                    if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                        if pp == 1: tp_scaling_data[profile_type][benchmark_key][engine][tp] = metric_value
                        if tp == 1: pp_scaling_data[profile_type][benchmark_key][engine][pp] = metric_value
    return tp_scaling_data, pp_scaling_data


def get_time_unit_factor(max_value: float) -> Tuple[str, float]:
    """Determines appropriate time unit and scaling factor."""
    if max_value <= 0: return "s", 1.0
    elif max_value < 1e-3: return "µs", 1e6
    elif max_value < 1: return "ms", 1e3
    else: return "s", 1.0

def assign_engine_styles(engines: List[str]) -> Dict[str, Dict]:
    """
    Assigns plotting styles dynamically, giving blues to 'vajra*' engines.
    Input 'engines' should be pre-sorted in the desired plotting order.
    """
    styles = {}
    hatch_cycle = itertools.cycle(PLOT_HATCHES)
    marker_cycle = itertools.cycle(PLOT_MARKERS)
    linestyle_cycle = itertools.cycle(PLOT_LINESTYLES)

    # Use separate color cycles based on the engine name prefix
    vajra_color_cycle = itertools.cycle(VAJRA_BLUES_PALETTE)
    other_color_cycle = itertools.cycle(OTHER_ENGINE_PALETTE)

    for engine in engines: # Iterate through the pre-sorted list
        is_vajra = engine.lower().startswith("vajra")
        color = next(vajra_color_cycle) if is_vajra else next(other_color_cycle)

        styles[engine] = {
            'color': color,
            'hatch': next(hatch_cycle),
            'marker': next(marker_cycle),
            'linestyle': next(linestyle_cycle),
        }
    return styles

def sort_engines(engine_list: List[str]) -> List[str]:
    """Sorts engines, placing 'vajra*' first, then alphabetically."""
    # Sort key: (0 if starts with 'vajra', 1 otherwise), engine_name
    return sorted(engine_list, key=lambda e: (0 if e.lower().startswith("vajra") else 1, e))

# --- Plotting Functions ---

def plot_grouped_comparison(
    tp_pp_key: str,
    profile_type: str,
    engine_data: Dict[str, Dict[str, float]],
    output_dir_path: Path,
    metric: MetricType,
    skip_engines: List[str]):
    """
    Generates comparison bar chart with vajra* first, internal legend, full frame,
    dynamic styles, and rotated multiplier text.
    """
    if not engine_data: return

    # Get engines present in this plot's data and sort them (vajra* first)
    present_engines_unsorted = list(engine_data.keys())
    present_engines = sort_engines(present_engines_unsorted) # Apply custom sort
    if not present_engines: return

    # Assign styles based on the *sorted* order
    engine_styles = assign_engine_styles(present_engines)

    reference_engine = present_engines[0] # First engine in the sorted list is reference
    all_benchmark_keys_unsorted = set().union(*(data.keys() for data in engine_data.values()))
    if not all_benchmark_keys_unsorted: return

    profile_type_simple = profile_type.split('veeksha_')[-1]
    all_benchmark_keys = sorted(list(all_benchmark_keys_unsorted), key=lambda k: parse_benchmark_key(k, profile_type))

    max_val = 0.0
    for eng in present_engines: # Iterate sorted list
        for val in engine_data[eng].values():
            if isinstance(val, (int, float)) and not math.isnan(val):
                 max_val = max(max_val, val)
    time_unit, time_factor = get_time_unit_factor(max_val)

    reference_values_orig = {
         key: engine_data[reference_engine].get(key, np.nan)
         for key in all_benchmark_keys
    }

    n_engines = len(present_engines); n_benchmarks = len(all_benchmark_keys)
    total_group_width = 0.8
    bar_width = total_group_width / n_engines
    fig_width = max(8, n_benchmarks * n_engines * 0.55)
    fig_height = 5.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    metric_name_capitalized = metric.capitalize()
    x = np.arange(n_benchmarks)
    plotted_bars = {}

    for i, engine in enumerate(present_engines): # Iterate through the sorted engines
        if engine in skip_engines: continue
        style = engine_styles[engine]
        values_scaled = []
        values_orig_for_calc = []
        for key in all_benchmark_keys:
            val_orig = engine_data[engine].get(key, np.nan)
            values_orig_for_calc.append(val_orig)
            values_scaled.append(val_orig * time_factor if isinstance(val_orig, (int, float)) and not math.isnan(val_orig) else np.nan)

        bar_positions = x - total_group_width / 2 + i * bar_width + bar_width / 2

        bars = ax.bar(bar_positions, values_scaled, bar_width, label=engine,
                      color=style['color'], alpha=PLOT_OPACITY, hatch=style['hatch'],
                      edgecolor='black', linewidth=0.8)
        plotted_bars[engine] = bars

        # Add multiplier text (Rotated)
        current_ylim = ax.get_ylim()
        yrange = current_ylim[1] - current_ylim[0]
        text_offset = yrange * 0.015 if yrange > 0 else 0.01 # Handle zero range

        for bar_idx, bar in enumerate(bars):
            key = all_benchmark_keys[bar_idx]
            current_value_scaled = bar.get_height()
            current_value_orig = values_orig_for_calc[bar_idx]
            ref_value_orig = reference_values_orig.get(key, np.nan)
            multiplier_text = ""

            if not np.isnan(current_value_orig) and current_value_orig > 1e-12:
                if engine == reference_engine: multiplier_text = "1.0x"
                elif not np.isnan(ref_value_orig) and ref_value_orig > 1e-12:
                    multiplier = current_value_orig / ref_value_orig
                    multiplier_text = f"{multiplier:.1f}x"
                else: multiplier_text = "N/A"

                text_y = current_value_scaled + text_offset
                if yrange > 0 and (text_y - current_ylim[0]) / yrange > 1.0 :
                     text_y = current_ylim[1] * 0.995

                ax.text(bar.get_x() + bar.get_width() / 2., text_y, multiplier_text,
                        rotation=90, ha='center', va='bottom',
                        fontsize=MULTIPLIER_FONTSIZE, fontweight='medium')

    # --- Plot Formatting ---
    y_label = f'{metric_name_capitalized} Latency ({time_unit})'
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
    ax.set_xticks(x)
    ax.set_xticklabels(all_benchmark_keys, rotation=30, ha='right', fontsize=TICK_LEGEND_FONTSIZE)
    profile_display_name = profile_type_simple.capitalize()
    title = f'{profile_display_name} {metric_name_capitalized} Latency ({tp_pp_key})'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15, fontweight=LABEL_FONTWEIGHT)
    x_axis_label = 'Benchmark Key'
    if profile_type_simple == 'prefill': x_axis_label = 'Prefill Length (tokens)'
    elif profile_type_simple == 'decode': x_axis_label = 'Context Length_Batch Size'
    ax.set_xlabel(x_axis_label, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)

    # --- Internal Legend (using sorted engine order) ---
    handles = [plotted_bars[eng] for eng in present_engines if eng in plotted_bars]
    labels = [eng for eng in present_engines if eng in plotted_bars]
    ax.legend(handles, labels, fontsize=TICK_LEGEND_FONTSIZE, loc='best',
              frameon=True, title="Engine", title_fontsize=TICK_LEGEND_FONTSIZE)

    ax.grid(True, linestyle="--", axis='y', alpha=0.6, color='darkgrey', linewidth=0.7)
    ax.set_axisbelow(True)

    # Log Scale Logic
    all_positive_scaled_values = [val_s for eng in present_engines for val_orig in engine_data[eng].values() if not np.isnan(val_orig) and val_orig > 1e-15 and (val_s := val_orig * time_factor) > 1e-9]
    if all_positive_scaled_values:
        min_val_s, max_val_s = min(all_positive_scaled_values), max(all_positive_scaled_values)
        if max_val_s / min_val_s > 50:
             ax.set_yscale('log')
             y_label_log = f'{metric_name_capitalized} Latency ({time_unit}) (log scale)'
             ax.set_ylabel(y_label_log, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
             current_ylim = ax.get_ylim(); ax.set_ylim(current_ylim[0], current_ylim[1] ** 1.05)

    ax.tick_params(axis='both', which='major', labelsize=TICK_LEGEND_FONTSIZE)
    # Keep full frame (no spine removal)

    # --- Save Plot ---
    base_plot_filename = f"{profile_type}_{tp_pp_key}_{metric}_comparison"
    png_filepath = output_dir_path / f"{base_plot_filename}.png"
    pdf_filepath = output_dir_path / f"{base_plot_filename}.pdf"
    try:
        fig.tight_layout()
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_filepath, bbox_inches='tight')
        print(f"  Saved grouped plots: {png_filepath.name}, {pdf_filepath.name}")
    except Exception as e: print(f"  Error saving grouped plot {base_plot_filename}: {e}")
    plt.close(fig)


def plot_scaling(
    scaling_type: str,
    profile_type: str,
    benchmark_key: str,
    engine_scaling_data: Dict[str, Dict[int, float]],
    output_dir_path: Path,
    metric: MetricType,
    skip_engines: List[str]):
    """
    Generates scaling line plot with vajra* first, internal legend, full frame,
    and dynamic styles.
    """
    if not engine_scaling_data: return

    present_engines_unsorted = list(engine_scaling_data.keys())
    present_engines = sort_engines(present_engines_unsorted) # Apply custom sort
    if not present_engines: return

    engine_styles = assign_engine_styles(present_engines) # Assign styles based on sorted order

    max_val = 0.0
    for eng in present_engines:
        for val in engine_scaling_data[eng].values():
             if isinstance(val, (int, float)) and not math.isnan(val): max_val = max(max_val, val)
    time_unit, time_factor = get_time_unit_factor(max_val)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    all_dim_values = set()
    metric_name_capitalized = metric.capitalize()
    profile_type_simple = profile_type.split('veeksha_')[-1]
    plotted_lines = {}

    for i, engine in enumerate(present_engines): # Iterate through sorted engines
        if engine in skip_engines: continue
        style = engine_styles[engine]
        scaling_points = engine_scaling_data.get(engine, {})
        valid_points = {dim: val for dim, val in scaling_points.items() if isinstance(val, (int, float)) and not math.isnan(val)}
        if not valid_points: continue
        sorted_points = sorted(valid_points.items())
        if len(sorted_points) < 1: continue

        dim_values = [p[0] for p in sorted_points]
        values_scaled = [p[1] * time_factor for p in sorted_points]
        all_dim_values.update(dim_values)

        line, = ax.plot(dim_values, values_scaled, marker=style['marker'], linestyle=style['linestyle'],
                        label=engine, color=style['color'],
                        markersize=6, linewidth=1.8, alpha=PLOT_OPACITY)
        plotted_lines[engine] = line

    if not all_dim_values: plt.close(fig); return

    # --- Plot Formatting ---
    x_label = f"{'Tensor' if scaling_type == 'TP' else 'Pipeline'} Parallelism ({scaling_type})"
    y_label = f'{metric_name_capitalized} Latency ({time_unit})'
    ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)
    profile_display_name = profile_type_simple.capitalize()
    title = f'{profile_display_name} {metric_name_capitalized} Scaling vs {scaling_type} ({benchmark_key})'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=15, fontweight=LABEL_FONTWEIGHT)

    # --- Internal Legend (using sorted engine order) ---
    handles = [plotted_lines[eng] for eng in present_engines if eng in plotted_lines]
    labels = [eng for eng in present_engines if eng in plotted_lines]
    ax.legend(handles, labels, fontsize=TICK_LEGEND_FONTSIZE, loc='best',
              frameon=True, title="Engine", title_fontsize=TICK_LEGEND_FONTSIZE)

    ax.grid(True, linestyle="--", axis='both', alpha=0.6, color='darkgrey', linewidth=0.7)
    ax.set_axisbelow(True)

    # X Ticks Logic
    sorted_dim_values = sorted(list(all_dim_values))
    int_dim_values = sorted([int(d) for d in sorted_dim_values if d == int(d)])
    if len(int_dim_values) > 1:
        max_dim = max(int_dim_values); potential_ticks = list(range(1, max_dim + 1))
        shown_ticks = sorted(list(set(potential_ticks) & set(int_dim_values)))
        if not shown_ticks and int_dim_values: shown_ticks = int_dim_values
        elif int_dim_values:
            if int_dim_values[0] not in shown_ticks: shown_ticks.insert(0, int_dim_values[0])
            if int_dim_values[-1] not in shown_ticks: shown_ticks.append(int_dim_values[-1])
        shown_ticks = sorted(list(set(shown_ticks)))
        ax.set_xticks(shown_ticks); ax.set_xticklabels(shown_ticks)
    else: ax.set_xticks(sorted_dim_values); ax.set_xticklabels(sorted_dim_values)

    # Log Scale Logic
    all_positive_scaled_values = [val_s for eng in present_engines for val_orig in engine_scaling_data[eng].values() if not np.isnan(val_orig) and val_orig > 1e-15 and (val_s := val_orig*time_factor) > 1e-9]
    if all_positive_scaled_values:
         min_val_s, max_val_s = min(all_positive_scaled_values), max(all_positive_scaled_values)
         if max_val_s / min_val_s > 30:
             ax.set_yscale('log')
             y_label_log = f'{y_label} (log scale)'
             ax.set_ylabel(y_label_log, fontsize=LABEL_FONTSIZE, fontweight=LABEL_FONTWEIGHT)

    # --- Finish Plot ---
    ax.tick_params(axis='both', which='major', labelsize=TICK_LEGEND_FONTSIZE)
    # Keep full frame (no spine removal)

    # --- Save Plot ---
    base_plot_filename = f"{profile_type}_{benchmark_key}_{scaling_type}_{metric}_scaling"
    png_filepath = output_dir_path / f"{base_plot_filename}.png"
    pdf_filepath = output_dir_path / f"{base_plot_filename}.pdf"
    try:
        fig.tight_layout()
        plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_filepath, bbox_inches='tight')
        print(f"  Saved scaling plots: {png_filepath.name}, {pdf_filepath.name}")
    except Exception as e: print(f"  Error saving scaling plot {base_plot_filename}: {e}")
    plt.close(fig)


# --- Main Execution --- (No changes needed)
def main():
    parser = argparse.ArgumentParser(description="Generate comparison and scaling plots from benchmark results.")
    parser.add_argument("results_dir", type=str, help="Path to the base directory containing engine result folders.")
    parser.add_argument("-o", "--output-dir", type=str, default="benchmark_plots", help="Base directory for plot subdirectories.")
    parser.add_argument("--metric", type=str, choices=["mean", "median"], default="median", help="Metric to plot (mean or median latency).")
    parser.add_argument("--skip-grouped", action="store_true", help="Skip grouped bar charts.")
    parser.add_argument("--skip-scaling", action="store_true", help="Skip scaling line plots.")
    parser.add_argument("--skip-engines", type=str, nargs="+", default=[], help="List of engine names to skip.")
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    if not results_path.is_dir(): print(f"Error: Results directory not found: {args.results_dir}"); return

    metric_output_suffix = f"_{args.metric}"
    base_output_path = Path(args.output_dir)
    grouped_output_dir_path = base_output_path / f"grouped_comparisons{metric_output_suffix}"
    scaling_output_dir_path = base_output_path / f"scaling_plots{metric_output_suffix}"

    if not args.skip_grouped: grouped_output_dir_path.mkdir(parents=True, exist_ok=True)
    if not args.skip_scaling: scaling_output_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving '{args.metric}' grouped plots to: {grouped_output_dir_path.resolve()}")
    print(f"Saving '{args.metric}' scaling plots to: {scaling_output_dir_path.resolve()}")

    parsed_data = find_and_parse_results(args.results_dir, args.metric)
    if not parsed_data: return

    if not args.skip_grouped:
        print(f"\n--- Generating Grouped Comparison Plots ({args.metric.capitalize()}) ---")
        sorted_tp_pp_keys = sorted(parsed_data.keys(), key=lambda k: tuple(map(int, re.findall(r'\d+', k))))
        plot_count = 0
        for tp_pp_key in sorted_tp_pp_keys:
            profile_data = parsed_data[tp_pp_key]
            sorted_profile_types = sorted(profile_data.keys())
            for profile_type_key in sorted_profile_types:
                engine_results = profile_data[profile_type_key]
                if engine_results:
                    plot_grouped_comparison(tp_pp_key, profile_type_key, engine_results, grouped_output_dir_path, args.metric, args.skip_engines)
                    plot_count += 1
        print(f"Generated {plot_count} grouped comparison plots (PNG+PDF).")
    else: print("\n--- Skipping Grouped Comparison Plots ---")

    if not args.skip_scaling:
        print(f"\n--- Generating Scaling Plots ({args.metric.capitalize()}) ---")
        tp_scaling_data, pp_scaling_data = restructure_for_scaling(parsed_data)
        plot_count_tp, plot_count_pp = 0, 0
        print("  Processing TP scaling (PP=1)...")
        for profile_type_key in sorted(tp_scaling_data.keys()):
             bench_data = tp_scaling_data[profile_type_key]
             for benchmark_key in sorted(bench_data.keys(), key=lambda k: parse_benchmark_key(k, profile_type_key)):
                 engine_data = bench_data[benchmark_key]
                 if any(sum(1 for v in points.values() if isinstance(v,(int,float)) and not math.isnan(v)) > 1 for points in engine_data.values()):
                     plot_scaling("TP", profile_type_key, benchmark_key, engine_data, scaling_output_dir_path, args.metric, args.skip_engines); plot_count_tp += 1
        print(f"  Generated {plot_count_tp} TP scaling plots (PNG+PDF).")
        print("  Processing PP scaling (TP=1)...")
        for profile_type_key in sorted(pp_scaling_data.keys()):
             bench_data = pp_scaling_data[profile_type_key]
             for benchmark_key in sorted(bench_data.keys(), key=lambda k: parse_benchmark_key(k, profile_type_key)):
                 engine_data = bench_data[benchmark_key]
                 if any(sum(1 for v in points.values() if isinstance(v,(int,float)) and not math.isnan(v)) > 1 for points in engine_data.values()):
                     plot_scaling("PP", profile_type_key, benchmark_key, engine_data, scaling_output_dir_path, args.metric, args.skip_engines); plot_count_pp += 1
        print(f"  Generated {plot_count_pp} PP scaling plots (PNG+PDF).")
    else: print("\n--- Skipping Scaling Plots ---")

    print("\nPlot generation finished.")

if __name__ == "__main__":
    main()
