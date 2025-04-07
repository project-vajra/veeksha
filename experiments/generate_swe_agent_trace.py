import json
import time
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse
import os
import sys

# --- Configuration ---
HF_DATASET_NAME = "nebius/SWE-agent-trajectories"
HF_DATASET_SPLIT = "train" # Or 'dev', 'test'
OUTPUT_FILENAME = "swe_agent_mooncake_style_trace.jsonl"

# Timestamp Simulation Parameters
# Delay BETWEEN different trajectories (simulates gap between sessions)
DELAY_BETWEEN_TRAJECTORIES_MS = (2000, 10000) # Wider range for inter-session gaps

# Threshold to filter out very large IATs from MOONCAKE trace
# (assuming >15s is likely a gap between sessions, not turns)
MOONCAKE_IAT_UPPER_BOUND_MS = 15000

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Convert SWE-Agent trajectories to Mooncake-style trace with realistic delays.')
    parser.add_argument('--mooncake-trace', required=True, help='Path to the MOONCAKE trace file (.jsonl) to sample inter-turn delays from.')
    parser.add_argument('--output-file', default=OUTPUT_FILENAME, help=f'Output filename for the converted trace (default: {OUTPUT_FILENAME})')
    parser.add_argument('--hf-dataset', default=HF_DATASET_NAME, help=f'Hugging Face dataset name (default: {HF_DATASET_NAME})')
    parser.add_argument('--hf-split', default=HF_DATASET_SPLIT, help=f'Dataset split to use (default: {HF_DATASET_SPLIT})')
    parser.add_argument('--iat-upper-bound', type=int, default=MOONCAKE_IAT_UPPER_BOUND_MS, help='Upper bound (ms) to filter Mooncake IATs for inter-turn delay sampling.')

    return parser.parse_args()

# --- Helper Functions ---
def simulate_inter_trajectory_delay():
    """Simulates a delay BETWEEN trajectories in seconds."""
    return random.uniform(DELAY_BETWEEN_TRAJECTORIES_MS[0] / 1000.0, DELAY_BETWEEN_TRAJECTORIES_MS[1] / 1000.0)

import json
import pandas as pd
import numpy as np
import sys
import os # Added for basename

def get_prefix_match_length(list1, list2):
    """Calculates the length of the common prefix between two lists."""
    match_len = 0
    min_len = min(len(list1), len(list2))
    for i in range(min_len):
        if list1[i] == list2[i]:
            match_len += 1
        else:
            break
    return match_len

def load_and_extract_mooncake_delays_session_aware(filepath, upper_bound_ms):
    """
    Loads a Mooncake trace, identifies requests within the same 'session'
    (based on hash_id prefix match with the previous request), calculates
    the Inter-Arrival Time (IAT) ONLY for these intra-session requests,
    filters them, and returns a list of valid delays in milliseconds.
    """
    print(f"Loading Mooncake trace for session-aware delay sampling: {filepath}")
    data = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    # Ensure hash_ids is loaded as a list
                    record = json.loads(line.strip())
                    if 'hash_ids' in record and not isinstance(record['hash_ids'], list):
                         # Attempt to fix if it's a string representation of a list, etc.
                         # This part might need adjustment based on actual data issues
                         try:
                            # A simple eval might work for basic cases, but BE CAREFUL with untrusted data
                            # record['hash_ids'] = eval(record['hash_ids'])
                            # A safer approach might be needed depending on format
                             pass # Or add safer parsing logic if needed
                         except Exception:
                            record['hash_ids'] = [] # Default to empty if parsing fails
                    elif 'hash_ids' not in record:
                        record['hash_ids'] = [] # Ensure key exists

                    data.append(record)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {filepath}", file=sys.stderr)
    except FileNotFoundError:
        print(f"Error: Mooncake trace file '{filepath}' not found.", file=sys.stderr)
        return None

    if not data:
        print(f"No valid data loaded from Mooncake trace {filepath}.", file=sys.stderr)
        return None

    df = pd.DataFrame(data)
    if 'timestamp' not in df.columns or 'hash_ids' not in df.columns:
        print(f"Error: Mooncake trace file {filepath} lacks 'timestamp' or 'hash_ids' column.", file=sys.stderr)
        return None

    # Ensure hash_ids column contains lists (handling potential loading issues)
    df['hash_ids'] = df['hash_ids'].apply(lambda x: x if isinstance(x, list) else [])

    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    intra_session_delays_ms = []
    print("Identifying intra-session delays based on hash prefix matches...")

    # Iterate through requests starting from the second one
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        previous_row = df.iloc[i-1]

        current_hashes = current_row['hash_ids']
        previous_hashes = previous_row['hash_ids']
        current_ts = current_row['timestamp']
        previous_ts = previous_row['timestamp']

        # Check if they belong to the same session (share a prefix)
        match_len = get_prefix_match_length(current_hashes, previous_hashes)

        if match_len > 0:
            # This request is likely a follow-up in the same session
            delay = current_ts - previous_ts
            # Basic filtering for valid delays
            if delay > 0:
                 intra_session_delays_ms.append(delay)

    if not intra_session_delays_ms:
         print(f"Warning: No consecutive requests with matching hash prefixes found in {filepath}.", file=sys.stderr)
         print("Cannot extract session-based delays. Will fall back to default uniform delay simulation.", file=sys.stderr)
         return None # Indicate fallback

    # Apply the upper bound filter
    filtered_delays = [d for d in intra_session_delays_ms if d <= upper_bound_ms]

    if not filtered_delays:
        print(f"Warning: No intra-session delays found below the upper bound of {upper_bound_ms}ms.", file=sys.stderr)
        print(f"  (Found {len(intra_session_delays_ms)} delays before filtering). Will fall back.", file=sys.stderr)
        return None # Indicate fallback


    print(f"Extracted {len(filtered_delays)} valid *intra-session* delays (0ms < delay <= {upper_bound_ms}ms) from {os.path.basename(filepath)}.")
    if filtered_delays: # Avoid errors if list becomes empty after filtering
        print(f"  Stats (filtered): Min={min(filtered_delays):.2f}ms, Max={max(filtered_delays):.2f}ms, Mean={np.mean(filtered_delays):.2f}ms, Median={np.median(filtered_delays):.2f}ms")
    return filtered_delays

# --- Main Processing Logic ---
def process_trajectories(args, mooncake_inter_turn_delays_ms):
    print(f"Loading dataset: {args.hf_dataset}, split: {args.hf_split}...")
    try:
        dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        print("Please ensure you have internet access and the 'datasets' library installed (`pip install datasets`).", file=sys.stderr)
        return

    print("Dataset loaded. Starting processing...")

    overall_timestamp_sec = 0.0
    records_written = 0
    fallback_delay_used = 0
    total_turns_simulated = 0

    # --- Define the fallback delay function ---
    # Use a plausible default range if Mooncake delays aren't available
    DEFAULT_DELAY_BETWEEN_TURNS_MS = (200, 1500)
    def get_inter_turn_delay_sec():
        nonlocal fallback_delay_used, total_turns_simulated
        total_turns_simulated += 1
        if mooncake_inter_turn_delays_ms:
            # Sample from Mooncake IATs
            delay_ms = random.choice(mooncake_inter_turn_delays_ms)
            # Ensure delay is not zero to prevent timestamp issues
            return max(delay_ms / 1000.0, 0.001) # Convert to seconds, ensure min 1ms
        else:
            # Fallback to uniform distribution
            fallback_delay_used += 1
            delay_ms = random.uniform(*DEFAULT_DELAY_BETWEEN_TURNS_MS)
            return delay_ms / 1000.0 # Convert to seconds

    with open(args.output_file, 'w') as outfile:
        for idx, item in enumerate(dataset):
            trajectory = item.get('trajectory')
            if not trajectory or not isinstance(trajectory, list):
                print(f"Skipping item {idx}: Invalid or missing trajectory.")
                continue

            # print(f"Processing trajectory {idx+1}/{len(dataset)}...") # Verbose

            # Simulate delay BETWEEN trajectories
            if idx > 0:
                 overall_timestamp_sec += simulate_inter_trajectory_delay()

            last_user_observation = ""
            last_ai_output = ""
            initial_input_accumulated = ""

            for turn_index, turn in enumerate(trajectory):
                role = turn.get('role')
                text = turn.get('text') or "" # Handle None text

                input_len = 0
                output_len = 0
                record = None

                if role == 'system':
                    initial_input_accumulated += text + "\n"
                    continue # Part of setup

                elif role == 'user':
                    # First user turn is the issue description - combined with system prompt
                    if turn_index == 1:
                         initial_input_accumulated += text + "\n"
                         last_user_observation = initial_input_accumulated # Input for first AI
                         continue # Not an interaction step generating trace output yet

                    # Subsequent user turns are environment observations
                    else:
                         input_len = len(last_ai_output) # Input was the last AI output
                         output_len = len(text)         # Output is the observation
                         last_user_observation = text   # Store for next AI turn

                         # Simulate timestamp for receiving the observation
                         overall_timestamp_sec += get_inter_turn_delay_sec()
                         record = {
                             "timestamp": int(overall_timestamp_sec * 1000), # Convert to ms
                             "input_length": input_len,
                             "output_length": output_len,
                         }

                elif role == 'ai':
                    # Agent's turn (reasoning + command)
                    input_len = len(last_user_observation) # Input was the last observation
                    output_len = len(text)                # Output is the agent's response
                    last_ai_output = text                 # Store this AI output

                    last_user_observation = "" # Reset context for next AI turn

                    # Simulate timestamp for the AI generating its response
                    overall_timestamp_sec += get_inter_turn_delay_sec()
                    record = {
                         "timestamp": int(overall_timestamp_sec * 1000), # Convert to ms
                         "input_length": input_len,
                         "output_length": output_len,
                     }

                # Write the record if one was generated for this turn
                if record is not None and (record["input_length"] > 0 or record["output_length"] > 0):
                    try:
                        outfile.write(json.dumps(record) + '\n')
                        records_written += 1
                    except TypeError as e:
                        print(f"Error writing record: {e}. Record data: {record}", file=sys.stderr)


            # Add a small final delay after the last turn of a trajectory
            overall_timestamp_sec += get_inter_turn_delay_sec() / 2.0 # Shorter delay after last action

            if (idx + 1) % 500 == 0: # Print progress less often
                 print(f"  ... processed {idx+1} trajectories, written {records_written} records.")


    print("-" * 30)
    print(f"Processing complete.")
    print(f"Total records written: {records_written}")
    print(f"Output saved to: {args.output_file}")
    print(f"Final simulated timestamp (ms): {int(overall_timestamp_sec * 1000)}")
    if fallback_delay_used > 0:
        print(f"Warning: Used fallback uniform delay simulation for {fallback_delay_used}/{total_turns_simulated} turns ({fallback_delay_used/total_turns_simulated*100:.1f}%) due to lack of Mooncake delays.", file=sys.stderr)
    print("-" * 30)

# --- Run the script ---
if __name__ == "__main__":
    args = parse_args()

    # Load delays from Mooncake trace FIRST
    mooncake_delays = load_and_extract_mooncake_delays(args.mooncake_trace, args.iat_upper_bound)

    # Start processing SWE Agent trajectories
    start_time = time.time()
    process_trajectories(args, mooncake_delays) # Pass delays to the function
    end_time = time.time()
    print(f"Script execution time: {end_time - start_time:.2f} seconds")