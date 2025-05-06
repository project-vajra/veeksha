import json
import matplotlib.pyplot as plt
import numpy as np

def plot_input_length_distribution(jsonl_file_path, output_image_path):
    """
    Reads a JSONL file, extracts 'input_length' from each line,
    and plots the distribution as a histogram.

    Args:
        jsonl_file_path (str): Path to the input JSONL file.
        output_image_path (str): Path to save the output plot image.
    """
    input_lengths = []
    try:
        with open(jsonl_file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'input_length' in data:
                        input_lengths.append(data['input_length'])
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_file_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if not input_lengths:
        print("No 'input_length' data found in the file.")
        return

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(input_lengths, bins=50, color='skyblue', edgecolor='black') # Adjust bin count as needed
    plt.title('Distribution of Input Lengths')
    plt.xlabel('Input Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Add some statistics
    mean_len = np.mean(input_lengths)
    median_len = np.median(input_lengths)
    std_dev = np.std(input_lengths)
    plt.axvline(mean_len, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_len:.2f}')
    plt.axvline(median_len, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_len:.2f}')
    plt.legend()
    print(f"Input Length Stats: Mean={mean_len:.2f}, Median={median_len:.2f}, StdDev={std_dev:.2f}, Count={len(input_lengths)}")


    # Save the plot
    try:
        plt.savefig(output_image_path)
        print(f"Plot saved to {output_image_path}")
        # plt.show() # Uncomment if you want to display the plot interactively
    except Exception as e:
        print(f"Error saving plot: {e}")

if __name__ == "__main__":
    # Define the input file path based on the user's open file
    file_path = "/scratch/chus/repos/veeksha/data/generated_traces/swe_agent_trace_short.jsonl/0.5/sampled_trace_dr0.5_mmt0.25.jsonl"
    # Define the output image path
    output_path = "/scratch/chus/repos/veeksha/input_length_distribution.png"
    plot_input_length_distribution(file_path, output_path)
