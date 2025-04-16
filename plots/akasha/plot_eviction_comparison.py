import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

# Assuming constants.py defines FONT, otherwise define it here
try:
    from constants import FONT
except ImportError:
    print("Warning: constants.py not found or FONT not defined. Using 'serif'.")
    FONT = 'serif' # Fallback font

# Set global font properties consistent with the request
plt.rcParams.update({'font.size': 10}) # Base size
plt.rcParams.update({'font.family': FONT})
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'


def plot_eviction_comparison( # Renamed function
    sota_eviction_data=None,
    recomp_cost_data=None,
    hit_prob_data=None,
    our_eviction_data=None,
    num_blocks_override=None, # Allow overriding num_blocks if data is not provided
    output_dir='./output',
    output_filename='eviction_comparison', # New filename
    seq_len_kb=128, # Info for default num_blocks calculation
    chunk_size_kb=2   # Info for default num_blocks calculation and tick labels
    ):
    """
    Create a 2x2 grid visualizing eviction-related factors for an SOSP paper.
    Final Spaced Version: Increased vertical spacing (hspace) between rows
        to prevent overlap, based on the previous final version.
    """
    # --- Determine Number of Blocks ---
    if chunk_size_kb <= 0:
        raise ValueError("chunk_size_kb must be positive.")
    default_num_blocks = int(seq_len_kb / chunk_size_kb)
    num_blocks = default_num_blocks
    data_provided = False
    data_arrays = [sota_eviction_data, recomp_cost_data, hit_prob_data, our_eviction_data]
    for data in data_arrays:
        if data is not None:
            num_blocks = len(data)
            data_provided = True
            break
    if not data_provided and num_blocks_override is not None:
        num_blocks = num_blocks_override

    print(f"Plotting with num_blocks = {num_blocks} (Chunk size: {chunk_size_kb} KB)")

    # --- Data Generation (Synthetic - Replace with Real Data) ---
    if not data_provided:
        print(f"Generating synthetic data for {num_blocks} blocks...")
        sota_eviction_data = np.arange(1, num_blocks + 1)
        recomp_cost_data = np.power(np.linspace(0, 10, num_blocks), 2) + 1
        hit_prob_data = np.exp(-0.05 * np.arange(num_blocks)) + 0.05
        utility = recomp_cost_data * hit_prob_data
        our_eviction_ranks = np.empty_like(utility)
        our_eviction_ranks[np.argsort(utility)] = np.arange(1, num_blocks + 1)
        our_eviction_data = our_eviction_ranks

    # --- Basic Validation ---
    final_data = [sota_eviction_data, recomp_cost_data, hit_prob_data, our_eviction_data]
    if any(d is None for d in final_data):
        raise ValueError("One or more data arrays are missing after data loading/generation.")
    if not all(len(d) == num_blocks for d in final_data):
        raise ValueError(f"All data arrays must have the same length ({num_blocks}). Mismatched lengths found.")

    # --- Plotting Setup ---
    # Increase figure height slightly to accommodate spacing
    fig, axs = plt.subplots(2, 2, figsize=(3.33, 1.3)) # Increased height from 1.2 -> 1.5
    axs = axs.flatten()

    # --- Colormap and Normalization ---
    cmap = cm.get_cmap('plasma')
    cmap_r = cm.get_cmap('plasma_r')
    norm_rank = mcolors.Normalize(vmin=1, vmax=num_blocks)
    norm_cost = mcolors.Normalize(vmin=np.min(recomp_cost_data), vmax=np.max(recomp_cost_data))
    norm_prob = mcolors.Normalize(vmin=np.min(hit_prob_data), vmax=np.max(hit_prob_data))

    # Reshape data for imshow
    data_sota = np.array(sota_eviction_data).reshape(1, -1)
    data_cost = np.array(recomp_cost_data).reshape(1, -1)
    data_prob = np.array(hit_prob_data).reshape(1, -1)
    data_our = np.array(our_eviction_data).reshape(1, -1)

    # --- Plotting Gradients ---
    extent = [-0.5, num_blocks - 0.5, 0, 1]
    axs[0].imshow(data_sota, aspect='auto', cmap=cmap_r, norm=norm_rank, extent=extent)
    axs[1].imshow(data_prob, aspect='auto', cmap=cmap, norm=norm_prob, extent=extent)
    axs[2].imshow(data_cost, aspect='auto', cmap=cmap, norm=norm_cost, extent=extent)
    axs[3].imshow(data_our, aspect='auto', cmap=cmap_r, norm=norm_rank, extent=extent)

    # --- Titles / Subplot Labels ---
    title_y_pos = 1.18 # Adjusted y-position slightly higher if needed
    title_fontsize = 7
    axs[0].set_title('(a) LRU Eviction Order', fontsize=title_fontsize, y=title_y_pos, color='black', pad=-2) # Adjusted pad
    axs[1].set_title('(c) Hit Probability', fontsize=title_fontsize, y=title_y_pos, color='black', pad=-2)
    axs[2].set_title('(b) Recomputation Cost', fontsize=title_fontsize, y=title_y_pos, color='black', pad=-2)
    axs[3].set_title('(d) Heimdall Eviction Order', fontsize=title_fontsize, y=title_y_pos, color='black', pad=-2)

    # --- Configure Axes (Frames, Ticks with Token Counts) ---
    tick_fontsize = 5
    if num_blocks <= 1:
        tick_positions = [0]
    elif num_blocks <= 5:
        tick_positions = np.arange(num_blocks)
    else:
        tick_positions = np.linspace(0, num_blocks - 1, 5, dtype=int)

    tick_labels = [f'{int((pos + 1) * chunk_size_kb)}k' for pos in tick_positions]

    for i, ax in enumerate(axs):
        ax.set_yticks([])
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=tick_fontsize)
        ax.tick_params(axis='x', colors='black', length=2, pad=1) # Reduced padding further

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.set_frame_on(True)
        # reduce the thickness of the frames
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

    # --- Layout Adjustment ---
    # *** Increase hspace significantly to add vertical space between rows ***
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.12, top=0.80, # Adjust top/bottom margins for new height/hspace
                        wspace=0.15, hspace=0.95) # Increased hspace drastically from 0.6 -> 0.95


    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    plot_path_base = os.path.join(output_dir, output_filename)
    plt.savefig(f'{plot_path_base}.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.savefig(f'{plot_path_base}.png', bbox_inches='tight', dpi=300, pad_inches=0.01)
    plt.close(fig)
    print(f"Generated Final Spaced plot: {plot_path_base}.pdf / .png")

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running with default synthetic data (64 blocks, 2k chunk) ---")
    plot_eviction_comparison()