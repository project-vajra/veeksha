import argparse
import json
import matplotlib.pyplot as plt
import os

def visualize_prefill_stats(metrics_folder, engine):
    json_path = os.path.join(metrics_folder, 'prefill_stats.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON {json_path}: {e}")
        return

    # Extract groups
    groups = data.get('groups', {})
    if not groups:
        print(f"No 'groups' found in {json_path}.")
        return

    # Convert keys to integers and sort
    try:
        sorted_keys = sorted(groups.keys(), key=lambda x: int(x))
    except ValueError:
        print("Error: Could not list keys as integers for sorting.")
        return
    
    x_values = [int(k) for k in sorted_keys]
    means = [groups[k]['mean'] for k in sorted_keys]
    p90s = [groups[k]['p90'] for k in sorted_keys]
    p99s = [groups[k]['p99'] for k in sorted_keys]
    counts = [groups[k].get('count', 0) for k in sorted_keys]

    # Determine sample count text
    unique_counts = sorted(list(set(counts)))
    if len(unique_counts) == 1:
        count_text = f"n={unique_counts[0]} per prompt length"
    else:
        count_text = f"n={min(unique_counts)}-{max(unique_counts)} per prompt length"
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_values, means, marker='o', label='Mean TTFC', linestyle='-')
    plt.plot(x_values, p90s, marker='s', label='P90 TTFC', linestyle='--')
    plt.plot(x_values, p99s, marker='^', label='P99 TTFC', linestyle=':')
    
    plt.xlabel('Prompt Tokens')
    plt.ylabel('Time To First Token (s)')
    plt.title(f'Prefill Stats: TTFC vs Prompt Tokens ({engine})\n{count_text}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Set x-axis to log scale for better visualization of doubling token counts
    plt.xscale('log')
    plt.xticks(x_values, [str(x) for x in x_values])
    plt.minorticks_off()

    output_path = os.path.join(metrics_folder, 'prefill_stats_viz.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize prefill stats from a metrics folder.")
    parser.add_argument("metrics_folder", help="Path to the metrics folder containing prefill_stats.json")
    parser.add_argument("engine", help="Name of the engine (e.g. vllm, sglang)")
    args = parser.parse_args()
    
    visualize_prefill_stats(args.metrics_folder, args.engine)
