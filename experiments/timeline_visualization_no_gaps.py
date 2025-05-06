#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from constants import COLORS, HATCHES, OPACITY, MARKERS

def load_tasks(json_file):
    """Load tasks from a JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def plot_timeline(tasks, output_file=None, show_plot=True):
    """
    Plot a timeline visualization with no gaps between tiers.
    
    Args:
        tasks: List of task dictionaries with 'tier', 'start', 'duration', and 'label'
        output_file: Optional path to save the figure
        show_plot: Whether to display the plot
    """
    # Get unique tiers (rows) and sort them in a logical order
    tiers = sorted(list(set(task['tier'] for task in tasks)), 
                  key=lambda x: {'Recompute': 0, 'GPU': 1, 'CPU': 2, 'NVMe': 3}.get(x, 999))
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(14, len(tiers) * 0.8))  # Reduced height for tighter spacing
    
    # Define colors for different tiers using COLORS from constants.py
    colors = {
        'Recompute': COLORS[0],
        'GPU': COLORS[1],
        'CPU': COLORS[2],
        'NVMe': COLORS[3],
    }
    
    # Find the overall end time for the plot
    max_end_time = max(task['start'] + task['duration'] for task in tasks)
    
    # Plot each task as a rectangle
    for task in tasks:
        tier = task['tier']
        start = task['start']
        duration = task['duration']
        label = task.get('label', '')
        
        # Add the task rectangle with opacity from constants.py
        # No gaps between tiers - rectangles are full height
        tier_index = list(colors.keys()).index(tier) if tier in colors else 0
        rect = patches.Rectangle(
            (start, tiers.index(tier) - 0.5),  # Start from the bottom of the tier
            duration,
            1.0,  # Full height of tier
            linewidth=1,
            edgecolor='black',
            facecolor=colors.get(tier, 'gray'),
            alpha=OPACITY,
            # hatch=HATCHES[tier_index % len(HATCHES)]  # Uncomment for hatches
        )
        ax.add_patch(rect)
        
        # Add label text with special handling for CPU tier (smaller threshold and font)
        if tier == 'CPU':
            # For CPU, use smaller threshold and font size to fit in narrow chunks
            if duration > max_end_time * 0.01:  # Smaller threshold for CPU
                ax.text(
                    start + duration / 2,
                    tiers.index(tier),
                    f"{label}",  # Only show number for CPU to save space
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,  # Smaller font size for CPU
                    fontweight='bold'
                )
        elif duration > max_end_time * 0.03:
            ax.text(
                start + duration / 2,
                tiers.index(tier),
                f"{tier}.{label}",
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=10,  # Increased font size
                fontweight='bold'
            )

    ax.plot([0, max_end_time], [len(tiers) + 0.2, len(tiers) + 0.2], color='black', linestyle='-', linewidth=1.5)
    
    # Add small arrow markers at exactly the start and end points
    ax.plot(1.5, len(tiers) + 0.2, marker='<', color='black', markersize=6, markeredgewidth=1)
    ax.plot(max_end_time - 1.5, len(tiers) + 0.2, marker='>', color='black', markersize=6, markeredgewidth=1)
    
    # Add vertical dotted line at the end of the iteration
    ax.axvline(x=max_end_time, color='black', linestyle='dotted', linewidth=1.5)
    
    # Add the TTFT text above the line
    ax.text(max_end_time / 2, len(tiers) + 0.4, 'TTFT', 
            horizontalalignment='center', verticalalignment='bottom', fontsize=14)
    
    # Set up the axis
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers, fontsize=12)  # Increased font size
    ax.set_xlim(0, max_end_time * 1.05)
    ax.set_ylim(-0.5, len(tiers) + 0.5)
    
    # Add x-axis ticks to show time in steps of 50 ms
    tick_step = 50  # 50 ms steps
    max_tick = int(max_end_time / tick_step) * tick_step + tick_step
    tick_positions = np.arange(0, max_tick + tick_step, tick_step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{int(x)}" for x in tick_positions], fontsize=10)  # Increased font size
    ax.set_xlabel('Time (ms)', fontsize=12)  # Add x-axis label with ms unit
    
    # Remove the grid
    ax.grid(False)
    
    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the figure if an output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.savefig("chunk_timeline_no_gaps.pdf", dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    
    return fig, ax

if __name__ == "__main__":
    # Path to the tasks.json file
    tasks_file = os.path.join(os.path.dirname(__file__), "tasks.json")
    
    # Load tasks
    tasks = load_tasks(tasks_file)
    
    # Plot the timeline
    plot_timeline(tasks, output_file="timeline_visualization_no_gaps.png")
    
    print(f"Timeline visualization created and saved as 'timeline_visualization_no_gaps.png'")
