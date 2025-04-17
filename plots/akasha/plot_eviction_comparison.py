import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import json

# Assuming constants.py defines FONT, otherwise define it here
try:
    from constants import FONT
except ImportError:
    print("Warning: constants.py not found or FONT not defined. Using 'serif'.")
    FONT = 'serif' # Fallback font

# Set global font properties
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': FONT})
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'


def moving_average(data, window_size):
    """Applies a simple moving average with edge padding."""
    if window_size <= 1 or len(data) < window_size:
        return data
    pad_width = window_size // 2
    # Ensure data is numpy array for pad/convolve
    data = np.asarray(data)
    padded_data = np.pad(data, pad_width, mode='edge')
    weights = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, weights, mode='valid')
    # Ensure output length matches input length exactly
    if len(smoothed) != len(data):
        # This can happen with even window sizes and certain padding modes.
        # Fallback to original data if length mismatch occurs.
        print(f"Warning: Smoothed data length ({len(smoothed)}) mismatch input ({len(data)}). Using raw data.")
        return data
    return smoothed


def plot_eviction_comparison(
    json_file_path,
    chunk_size_kb=2,
    output_dir='./output',
    output_filename='eviction_comparison',
    smoothing_window_size=3,
    ):
    """
    Create a 2x2 grid visualizing eviction-related factors.
    Reads data from a JSON file, applying optional smoothing to recomputation cost.
    
    Data Requirements from JSON file:
    1. 'eviction_ordering': Heimdall eviction order
    2. 'per_chunk_times': Recomputation cost per chunk
    3. 'survival_prob': Dictionary of survival probabilities
    
    LRU eviction order is generated automatically if not present in the JSON.
    """
    # --- Data Acquisition ---
    loaded_data = {}
    num_blocks = None
    data_sources = {} # Track where data came from

    # Load from JSON file
    if not os.path.exists(json_file_path):
        raise ValueError(f"JSON file not found: {json_file_path}")
        
    print(f"Loading data from JSON file: {json_file_path}")
    try:
        with open(json_file_path, 'r') as f:
            loaded_data = json.load(f)
        print("  - JSON loaded successfully.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        raise

    # --- Extract data from JSON ---
    
    # 1. Heimdall Eviction Order ('eviction_ordering')
    if 'eviction_ordering' in loaded_data:
        eviction_order = loaded_data['eviction_ordering']
        # for every block 
        eviction_priority = [0] * len(eviction_order)
        for i, block in enumerate(eviction_order):
            # eviction_priority[block - 1] = len(eviction_priority) - i
            eviction_priority[block - 1] = i
        our_eviction_data = np.array(eviction_priority)
        print(f"  - Heimdall eviction order: {our_eviction_data}, eviction order: {eviction_order}")
        # we need to convert eviction order to priority for
        if num_blocks is None: num_blocks = len(our_eviction_data)
        data_sources['heimdall'] = f"JSON ('eviction_ordering', {len(our_eviction_data)} blocks)"
    else:
        raise ValueError("Missing 'eviction_ordering' in JSON file")

    # 2. Recomputation Cost ('per_chunk_times')
    if 'per_chunk_times' in loaded_data:
        times_dict = loaded_data['per_chunk_times']
        if num_blocks is None and times_dict:
             try:
                 # Infer num_blocks from max key if possible
                 current_max_key = 0
                 for k in times_dict.keys():
                     try:
                         current_max_key = max(current_max_key, int(k))
                     except ValueError: pass # Ignore non-integer keys
                 if current_max_key > 0:
                    num_blocks = current_max_key
                    print(f"  - Inferred num_blocks = {num_blocks} from 'per_chunk_times' keys.")
             except Exception as e_infer:
                  print(f"  - Warning: Error inferring num_blocks from 'per_chunk_times': {e_infer}")

        if num_blocks is not None:
            try:
                # Check if all keys from 1 to num_blocks exist
                expected_keys = {str(i) for i in range(1, num_blocks + 1)}
                if not expected_keys.issubset(times_dict.keys()):
                    raise KeyError(f"Missing one or more keys in range 1..{num_blocks} in 'per_chunk_times'")
                _recomp_cost_raw = np.array([times_dict[str(i)] for i in range(1, num_blocks + 1)])
                data_sources['recomp_cost'] = f"JSON ('per_chunk_times', {len(_recomp_cost_raw)} values)"
                # Length validation happens later, but ensure consistency if possible
                if len(_recomp_cost_raw) != num_blocks:
                     print(f"  - Warning: Length mismatch for 'per_chunk_times' ({len(_recomp_cost_raw)}) vs inferred num_blocks ({num_blocks}). Using length {len(_recomp_cost_raw)}.")
                     num_blocks = len(_recomp_cost_raw)
            except KeyError as e:
                 print(f"  - Error: {e}. Cannot load recomputation cost from JSON.")
                 raise
            except Exception as e:
                 print(f"  - Error processing 'per_chunk_times': {e}")
                 raise
    else:
        raise ValueError("Missing 'per_chunk_times' in JSON file")

    # 3. Survival Probability ('survival_prob')
    if 'survival_prob' in loaded_data:
        # Check if survival_prob is a dictionary (time-based probabilities)
        if isinstance(loaded_data['survival_prob'], dict):
            print("  - 'survival_prob' is a dictionary of 1-CDF values for request length fractions")
            # If we already know num_blocks, create an array of that size
            if num_blocks is not None:
                try:
                    # Convert dictionary keys to floats and sort them
                    sorted_keys = sorted([float(k) for k in loaded_data['survival_prob'].keys()])
                    # Get corresponding probability values
                    sorted_values = [loaded_data['survival_prob'][str(k)] for k in sorted_keys]
                    
                    # Create an interpolation function to map positions to probabilities
                    from scipy.interpolate import interp1d
                    
                    # Create interpolation function (linear interpolation)
                    interp_func = interp1d(sorted_keys, sorted_values, bounds_error=False, 
                                           fill_value=(sorted_values[0], sorted_values[-1]))
                    
                    # Generate positions for each block (0 to 1 range)
                    block_positions = np.linspace(0, 1, num_blocks)
                    
                    # Get interpolated probabilities for each block
                    hit_prob_data = interp_func(block_positions)
                    
                    data_sources['hit_prob'] = f"Interpolated from JSON ('survival_prob', {len(hit_prob_data)} values)"
                    print(f"  - Interpolated survival probabilities for {num_blocks} blocks from 'survival_prob'")
                except Exception as e:
                    print(f"  - Error processing survival_prob dictionary: {e}")
                    raise
            else:
                raise ValueError("Cannot determine num_blocks for interpolating 'survival_prob'")
        else:
            # If it's already an array-like object, use it directly
            hit_prob_data = np.array(loaded_data['survival_prob'])
            if num_blocks is None: num_blocks = len(hit_prob_data)
            data_sources['hit_prob'] = f"JSON ('survival_prob', {len(hit_prob_data)} values)"
    else:
        raise ValueError("Missing 'survival_prob' in JSON file")

    # 4. LRU Eviction Order (generate if not in JSON)
    if 'lru_eviction_order' in loaded_data:
        sota_eviction_data = np.array(loaded_data['lru_eviction_order'])
        data_sources['lru'] = f"JSON ('lru_eviction_order', {len(sota_eviction_data)} values)"
    elif num_blocks is not None:
        # Standard LRU eviction order: 1, 2, 3, ..., num_blocks
        sota_eviction_data = np.arange(1, num_blocks + 1)
        data_sources['lru'] = f"Synthetic (standard LRU 1..{num_blocks})"
        print(f"  - Generated synthetic LRU eviction order for {num_blocks} blocks")
    else:
        raise ValueError("Cannot determine num_blocks for generating LRU eviction order")

    # Print data sources clearly
    print("--- Data Sources ---")
    print(f"Heimdall Order:      {data_sources.get('heimdall', 'Not Loaded')}")
    print(f"Recomp Cost:         {data_sources.get('recomp_cost', 'Not Loaded')}")
    print(f"Survival Probability: {data_sources.get('hit_prob', 'Not Loaded')}")
    print(f"LRU Order:           {data_sources.get('lru', 'Not Loaded')}")
    print(f"Chunk Size:          {chunk_size_kb} KB")
    print("--------------------")

    print(f"Final num_blocks used for plotting = {num_blocks}")

    # --- Validate Data Components and Lengths ---
    final_data_map = {} # Store validated data
    
    # Add all components to the map
    final_data_map["LRU Order"] = sota_eviction_data
    final_data_map["Recomp Cost (Raw)"] = _recomp_cost_raw
    final_data_map["Survival Probability"] = hit_prob_data
    final_data_map["Heimdall Order"] = our_eviction_data

    # Validate chunk size
    if not isinstance(chunk_size_kb, (int, float)) or chunk_size_kb <= 0:
        raise ValueError(f"Chunk Size must be a positive number (got: {chunk_size_kb})")

    # Validate lengths against final num_blocks
    for name, data in final_data_map.items():
        if len(data) != num_blocks:
            raise ValueError(f"Length mismatch: {name} has length {len(data)}, expected {num_blocks}.")

    # --- Apply Smoothing to Recomputation Cost ---
    recomp_cost_data = final_data_map["Recomp Cost (Raw)"] # Start with validated raw data
    if smoothing_window_size > 1:
        smoothed_cost = moving_average(recomp_cost_data, smoothing_window_size)
        # Check if smoothing actually changed anything
        if not np.array_equal(smoothed_cost, recomp_cost_data):
             recomp_cost_data = smoothed_cost # Update with smoothed data
             print(f"Applied moving average smoothing (window={smoothing_window_size}) to recomputation cost.")
        elif num_blocks < smoothing_window_size:
             print(f"Skipping recomputation cost smoothing: num_blocks ({num_blocks}) < window_size ({smoothing_window_size}).")
        else:
             print(f"Note: Recomputation cost smoothing (window={smoothing_window_size}) was skipped or resulted in no change.")
    else:
        print("Skipping recomputation cost smoothing (window_size <= 1).")
    
    # Update map with final (potentially smoothed) cost data for plotting
    final_data_map["Recomp Cost"] = recomp_cost_data

    # --- Plotting Setup ---
    fig, axs = plt.subplots(2, 2, figsize=(3.33, 1.3))
    axs = axs.flatten()

    # --- Colormap and Normalization ---
    cmap_plasma = cm.get_cmap('plasma')
    cmap_plasma_r = cm.get_cmap('plasma_r') # Reversed for eviction rank

    norm_rank = mcolors.Normalize(vmin=1, vmax=max(1, num_blocks))

    # Use final cost data for normalization
    cost_min, cost_max = np.min(final_data_map["Recomp Cost"]), np.max(final_data_map["Recomp Cost"])
    prob_min, prob_max = np.min(final_data_map["Survival Probability"]), np.max(final_data_map["Survival Probability"])
    norm_cost = mcolors.Normalize(vmin=cost_min, vmax=max(cost_min + np.finfo(float).eps, cost_max))
    norm_prob = mcolors.Normalize(vmin=prob_min, vmax=max(prob_min + np.finfo(float).eps, prob_max))

    # Reshape data for imshow (needs 2D)
    data_sota = np.array(final_data_map["LRU Order"]).reshape(1, -1)
    data_cost = np.array(final_data_map["Recomp Cost"]).reshape(1, -1) # Use final (smoothed) cost
    data_prob = np.array(final_data_map["Survival Probability"]).reshape(1, -1)
    data_our = np.array(final_data_map["Heimdall Order"]).reshape(1, -1)

    # --- Plotting Gradients ---
    extent = [-0.5, num_blocks - 0.5, 0, 1]

    im_sota = axs[0].imshow(data_sota, aspect='auto', cmap=cmap_plasma_r, norm=norm_rank, extent=extent)
    im_prob = axs[1].imshow(data_prob, aspect='auto', cmap=cmap_plasma, norm=norm_prob, extent=extent)
    im_cost = axs[2].imshow(data_cost, aspect='auto', cmap=cmap_plasma, norm=norm_cost, extent=extent)
    im_our = axs[3].imshow(data_our, aspect='auto', cmap=cmap_plasma_r, norm=norm_rank, extent=extent)

    # --- Titles / Subplot Labels ---
    title_y_pos = 1.18
    title_fontsize = 7
    title_pad = -2
    axs[0].set_title('(a) LRU Eviction Order', fontsize=title_fontsize, y=title_y_pos, color='black', pad=title_pad)
    axs[1].set_title('(c) Survival Probability', fontsize=title_fontsize, y=title_y_pos, color='black', pad=title_pad)
    axs[2].set_title('(b) Recomputation Cost', fontsize=title_fontsize, y=title_y_pos, color='black', pad=title_pad)
    axs[3].set_title('(d) Heimdall Eviction Order', fontsize=title_fontsize, y=title_y_pos, color='black', pad=title_pad)

    # --- Configure Axes (Frames, Ticks with KB Labels) ---
    tick_fontsize = 5
    if num_blocks <= 1:
        tick_positions = [0]
    elif num_blocks <= 5:
        tick_positions = np.arange(num_blocks)
    else:
        num_ticks = min(5, num_blocks)
        tick_positions = np.linspace(0, num_blocks - 1, num_ticks, dtype=int)
        tick_positions[0] = 0
        tick_positions[-1] = num_blocks - 1
        tick_positions = np.unique(tick_positions)

    # Create KB labels (chunk_size_kb is now mandatory and validated)
    tick_labels = [f'{int((pos + 1) * chunk_size_kb)}k' for pos in tick_positions]

    for i, ax in enumerate(axs):
        ax.set_yticks([])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
        ax.tick_params(axis='x', colors='black', length=2, pad=1)

        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor('black')

    # --- Layout Adjustment ---
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.80,
                        wspace=0.15, hspace=0.95)

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path_base = os.path.join(output_dir, output_filename)
    try:
        plt.savefig(f'{plot_path_base}.pdf', bbox_inches='tight', pad_inches=0.01)
        plt.savefig(f'{plot_path_base}.png', bbox_inches='tight', dpi=300, pad_inches=0.01)
        print(f"Generated plot: {plot_path_base}.pdf / .png")
    except Exception as e:
        print(f"Error saving plots: {e}")
    finally:
        plt.close(fig)

# --- Example Usage ---
if __name__ == "__main__":
    plot_eviction_comparison(
        json_file_path='data/eviction_plot_data_h100_70b_tp8.json',
        chunk_size_kb=2,
        output_filename='eviction_comp_h100_70b_tp8',
        smoothing_window_size=3
    )
