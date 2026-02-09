
import argparse
import json
import matplotlib.pyplot as plt
import os
import glob
import math

def visualize_decode_stats(base_dir, engines):
    """
    Aggregates decode stats for multiple engines and plots comparisons.
    
    Expected directory structure:
    base_dir/
      decode_<context_length>/
        <engine>/
          sweep_<timestamp>/
            decode_stats.json
    """
    
    # Store aggregated data: {context_length: {engine: {batch_size: {mean: ..., p90: ..., p99: ..., count: ...}}}}
    aggregated_data = {}
    
    # Collect data for each engine
    for engine in engines:
        print(f"Collecting data for engine: {engine}")
        
        # Find all decode_stats.json files for this engine
        # Pattern: base_dir/decode_*/engine/*/decode_stats.json
        search_pattern = os.path.join(base_dir, "decode_*", engine, "*", "decode_stats.json")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"  No stats files found for {engine}")
            continue

        print(f"  Found {len(files)} files for {engine}")

        for file_path in files:
            # Extract context length from parent directory name (decode_1024 -> 1024)
            parts = file_path.split(os.sep)
            
            context_len = None
            for part in parts:
                if part.startswith("decode_") and part[7:].isdigit():
                    context_len = int(part[7:])
                    break
            
            if context_len is None:
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue

            if context_len not in aggregated_data:
                aggregated_data[context_len] = {}
            if engine not in aggregated_data[context_len]:
                aggregated_data[context_len][engine] = {}

            # Parse stats
            for key, stats in data.items():
                if key == "notes": continue
                
                try:
                    # Try to get actual batch size from config
                    if 'config' in stats and 'resolved_min_active_requests' in stats['config']:
                        batch_size = int(stats['config']['resolved_min_active_requests'])
                    else:
                        # Fallback to parsing key
                        _, batch_str = key.split('_')
                        batch_size = int(batch_str)
                    
                    tbc_stats = stats.get('tbc_in_window_stats', {})
                    if not tbc_stats:
                        continue
                        
                    metrics = {
                        'mean': tbc_stats.get('mean', 0) * 1000, # ms
                        'p90': tbc_stats.get('p90', 0) * 1000,
                        'p99': tbc_stats.get('p99', 0) * 1000,
                        'count': tbc_stats.get('count', 0)
                    }
                    
                    aggregated_data[context_len][engine][batch_size] = metrics
                    
                except (ValueError, KeyError) as e:
                    print(f"  Error parsing entry {key}: {e}")
                    continue

    if not aggregated_data:
        print("No valid data found to plot.")
        return

    context_lengths = sorted(aggregated_data.keys())
    num_ctx = len(context_lengths)
    
    # Determine grid size
    cols = 2
    rows = math.ceil(num_ctx / cols)
    
    # Styles config
    # Engine colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] 
    engine_colors = {eng: colors[i % len(colors)] for i, eng in enumerate(engines)}
    
    # Metric styles
    metrics_config = {
        'mean': {'style': '-', 'marker': 'o', 'label': 'Mean'},
        'p99': {'style': ':', 'marker': 's', 'label': 'P99'},
        'p90': {'style': '--', 'marker': '^', 'label': 'P90'} # Added P90 as requested
    }
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
    fig.suptitle('Decode TBC Comparison vs Batch Size (Mean, P90, P99)', fontsize=16)
    
    if num_ctx == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.flatten()
        
    for i, ctx_len in enumerate(context_lengths):
        ax = axes_flat[i]
        ctx_data = aggregated_data[ctx_len]
        
        lines_handles = []
        lines_labels = []
        
        # Plot for each engine
        for engine in engines:
            if engine not in ctx_data:
                continue
                
            engine_data = ctx_data[engine]
            sorted_batches = sorted(engine_data.keys())
            
            if not sorted_batches:
                continue
                
            color = engine_colors.get(engine, 'black')
            
            # Extract sample counts for legend
            counts = [engine_data[b]['count'] for b in sorted_batches]
            if counts:
                min_c, max_c = min(counts), max(counts)
                count_lbl = f"{min_c}" if min_c == max_c else f"{min_c}-{max_c}"
                engine_legend_label = f'{engine} (n={count_lbl})'
            else:
                engine_legend_label = engine
            
            # Plot dummy line for legend entry for engine color
            dummy, = ax.plot([], [], color=color, label=engine_legend_label)
            lines_handles.append(dummy)
            lines_labels.append(engine_legend_label)

            # Plot each metric
            for metric, style_cfg in metrics_config.items():
                y_values = [engine_data[b][metric] for b in sorted_batches]
                
                # Make lines slightly transparent to see overlap
                ax.plot(sorted_batches, y_values, 
                       linestyle=style_cfg['style'], 
                       marker=style_cfg['marker'],
                       color=color,
                       alpha=0.8)

        ax.set_title(f'Context: {ctx_len}')
        ax.set_xlabel('Batch Size (Actual)')
        ax.set_ylabel('TBC (ms)')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.set_xscale('log')
        
        # X-ticks logic
        all_batches = set()
        for eng_d in ctx_data.values():
             all_batches.update(eng_d.keys())
        if all_batches:
            sorted_all_batches = sorted(list(all_batches))
            ax.set_xticks(sorted_all_batches)
            ax.set_xticklabels([str(b) for b in sorted_all_batches], rotation=45, ha='right')
            
        # Custom legend for metrics
        # Add metric style guides to legend
        for metric, style_cfg in metrics_config.items():
            h, = ax.plot([], [], color='gray', linestyle=style_cfg['style'], marker=style_cfg['marker'], label=style_cfg['label'])
            lines_handles.append(h)
            lines_labels.append(style_cfg['label'])
            
        ax.legend(handles=lines_handles, labels=lines_labels, loc='upper left', fontsize='small')

    # Hide unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = 'decode_comparison_combined.png'
    output_path = os.path.join(base_dir, output_filename)
    plt.savefig(output_path)
    print(f"Saved combined comparison plot to {output_path}")
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize decode stats comparison.")
    parser.add_argument("base_dir", help="Base directory (e.g. nemotron_micro)")
    parser.add_argument("engines", nargs='+', help="List of engines (e.g. vllm sgl)")
    args = parser.parse_args()
    
    visualize_decode_stats(args.base_dir, args.engines)
