
import json
import time
import random
import math
from datasets import load_dataset, logging as ds_logging, IterableDataset
import argparse
import os
import sys
import concurrent.futures
# from functools import lru_cache # Keep commented unless needed within worker
from transformers import AutoTokenizer, logging as tfs_logging
from tqdm import tqdm
# import pickle # No longer needed

# Disable excessive logging
ds_logging.set_verbosity_error()
tfs_logging.set_verbosity_error()

# --- Configuration (Keep TOKENIZER_NAME global) ---
HF_DATASET_NAME = "nebius/SWE-agent-trajectories"
HF_DATASET_SPLIT = "train"
OUTPUT_FILENAME = "swe_agent_trace_overlap_large_map.jsonl"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"
MEAN_INTER_SESSION_START_TIME_MS = 250 # 250ms
MEAN_INTER_REQUEST_TIME_MS = 10000 # 10 seconds
BLOCK_SIZE = 512
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# --- Argument Parser (Unchanged) ---
def parse_args():
    parser = argparse.ArgumentParser(description='Convert SWE-Agent trajectories to trace format with overlapping sessions (large dataset optimized with map).')
    parser.add_argument('--output-file', default=OUTPUT_FILENAME, help=f'Output filename (default: {OUTPUT_FILENAME})')
    parser.add_argument('--hf-dataset', default=HF_DATASET_NAME, help=f'Hugging Face dataset name (default: {HF_DATASET_NAME})')
    parser.add_argument('--hf-split', default=HF_DATASET_SPLIT, help=f'Dataset split (default: {HF_DATASET_SPLIT})')
    parser.add_argument('--tokenizer', default=TOKENIZER_NAME, help=f'Tokenizer name (default: {TOKENIZER_NAME})')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help=f'Block size (default: {BLOCK_SIZE})')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help=f'Number of worker processes (default: {NUM_WORKERS})')
    parser.add_argument('--inter-session-start-rate', type=float, default=1000.0/MEAN_INTER_SESSION_START_TIME_MS if MEAN_INTER_SESSION_START_TIME_MS > 0 else float('inf'),
                        help=f'Rate of session starts (sessions/sec). Default corresponds to {MEAN_INTER_SESSION_START_TIME_MS}ms mean delay.')
    parser.add_argument('--inter-request-rate', type=float, default=1000.0/MEAN_INTER_REQUEST_TIME_MS if MEAN_INTER_REQUEST_TIME_MS > 0 else float('inf'),
                        help=f'Rate of intra-session requests (req/sec). Default corresponds to {MEAN_INTER_REQUEST_TIME_MS}ms mean delay.')
    parser.add_argument('--num-trajectories', type=int, default=None, help='Max number of trajectories to process (default: all)')
    parser.add_argument('--cache-dir', type=str, default=None, help='Hugging Face cache directory.')

    args = parser.parse_args()
    if args.inter_session_start_rate <= 0: parser.error("--inter-session-start-rate must be positive")
    if args.inter_request_rate <= 0: parser.error("--inter-request-rate must be positive")
    if args.block_size <= 0: parser.error("--block-size must be positive")
    if args.workers <= 0:
        print("Warning: --workers must be positive. Setting to 1.", file=sys.stderr)
        args.workers = 1

    args.mean_inter_session_start_delay_ms = 1000.0 / args.inter_session_start_rate if args.inter_session_start_rate != float('inf') else 0
    args.mean_inter_request_delay_ms = 1000.0 / args.inter_request_rate if args.inter_request_rate != float('inf') else 0

    return args

# --- Helper Functions (Unchanged, including get_tokenizer, tokenize_text, etc.) ---
def generate_poisson_delay(rate):
    if rate == float('inf'): return 0.0
    if rate <= 0: raise ValueError("Rate must be positive")
    delay_sec = random.expovariate(rate)
    return delay_sec * 1000.0

_tokenizer_cache = {}
def get_tokenizer(tokenizer_name, cache_dir=None):
    pid = os.getpid()
    if pid not in _tokenizer_cache:
        # print(f"[Worker {pid}] Loading tokenizer: {tokenizer_name}", file=sys.stderr) # Less verbose
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            _tokenizer_cache[pid] = tokenizer
        except Exception as e:
            print(f"[Worker {pid}] Error loading tokenizer '{tokenizer_name}': {e}", file=sys.stderr)
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e
    return _tokenizer_cache[pid]

def tokenize_text(text, tokenizer_name, cache_dir=None):
    if not text: return []
    try:
        tokenizer = get_tokenizer(tokenizer_name, cache_dir)
        return tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        # print(f"[Worker {os.getpid()}] Warning: Tokenization failed: '{str(text)[:50]}...'. E: {e}. Ret empty.", file=sys.stderr) # Less verbose
        return []

def chunk_and_hash_text(token_ids, block_size):
    if not token_ids or block_size <= 0: return []
    hash_ids = []
    for i in range(0, len(token_ids), block_size):
        block = tuple(token_ids[i:i+block_size])
        block_hash = abs(hash(block)) % (2**32)
        hash_ids.append(block_hash)
    return hash_ids

# --- Worker Function (Unchanged) ---
def process_trajectory_worker(trajectory_data, trajectory_idx, session_start_time_ms, args):
    """
    Worker function to process a single trajectory.
    Calculates absolute timestamps based on the provided session start time.
    Loads its own tokenizer instance.
    """
    entries = []
    # Ensure trajectory data is usable
    if isinstance(trajectory_data, dict) and 'trajectory' in trajectory_data:
        trajectory = trajectory_data['trajectory']
        instance_id = trajectory_data.get('instance_id', f"unknown_{trajectory_idx}")
    elif isinstance(trajectory_data, list): # Handle case where item is just the list
         trajectory = trajectory_data
         instance_id = f"unknown_{trajectory_idx}"
    else:
        return entries # Skip invalid data

    if not isinstance(trajectory, list) or not trajectory:
        return entries # Skip empty or invalid trajectory list

    current_prompt_text = ""
    request_count_in_session = 0
    requests_generated = [] # Store requests before assigning timestamps

    # --- First pass: Generate request details (input/output text) ---
    for turn_idx, turn in enumerate(trajectory):
        if not isinstance(turn, dict): continue
        role = turn.get('role', '').lower()
        text = turn.get('text', '')
        if not text: continue

        formatted_text = ""
        if role == 'system': formatted_text = f"System: {text}\n\n"
        elif role == 'user': formatted_text = f"User: {text}\n\n"
        elif role in ['ai', 'assistant', 'model']: formatted_text = f"Assistant: {text}\n\n"
        else: continue

        if role == 'user':
            input_prompt = current_prompt_text + formatted_text
            output_text = None
            assistant_turn_idx = -1
            for next_turn_idx in range(turn_idx + 1, len(trajectory)):
                next_turn = trajectory[next_turn_idx]
                next_role = next_turn.get('role', '').lower()
                if next_role in ['ai', 'assistant', 'model']:
                    assistant_text = next_turn.get('text')
                    if assistant_text:
                        output_text = f"Assistant: {assistant_text}\n\n"
                        assistant_turn_idx = next_turn_idx
                        break

            if output_text is not None:
                request_detail = {
                    'input_prompt': input_prompt,
                    'output_text': output_text,
                    'user_turn_id': turn_idx,
                    'assistant_turn_id': assistant_turn_idx,
                    'request_in_session': request_count_in_session
                }
                requests_generated.append(request_detail)
                request_count_in_session += 1
                current_prompt_text += formatted_text + output_text
            else:
                 current_prompt_text += formatted_text
        elif role == 'system':
             current_prompt_text += formatted_text

    if not requests_generated:
        return entries

    # --- Second pass: Tokenize, hash, and assign timestamps ---
    current_request_time_ms = session_start_time_ms
    for request_idx, req_detail in enumerate(requests_generated):
        # Ensure tokenizer is loaded before heavy processing
        # get_tokenizer(args.tokenizer, args.cache_dir) # Pre-load if causing issues, but usually loaded on demand

        input_tokens = tokenize_text(req_detail['input_prompt'], args.tokenizer, args.cache_dir)
        output_tokens = tokenize_text(req_detail['output_text'], args.tokenizer, args.cache_dir)
        hash_ids = chunk_and_hash_text(input_tokens, args.block_size)

        if request_idx == 0:
            timestamp = int(round(current_request_time_ms))
        else:
            inter_request_delay_ms = generate_poisson_delay(args.inter_request_rate)
            current_request_time_ms += inter_request_delay_ms
            timestamp = int(round(current_request_time_ms))

        trace_entry = {
            'timestamp': timestamp,
            'hash_ids': hash_ids,
            'input_length': len(input_tokens),
            'output_length': len(output_tokens),
            'trajectory_id': trajectory_idx,
            'instance_id': instance_id,
            'user_turn_id': req_detail['user_turn_id'],
            'assistant_turn_id': req_detail['assistant_turn_id'],
            'request_in_session': req_detail['request_in_session']
        }
        entries.append(trace_entry)

    return entries


# --- Helper function for map ---
def worker_wrapper(params):
    """Unpacks arguments and calls the main worker function. Handles exceptions."""
    trajectory_data, trajectory_idx, session_start_time_ms, args = params
    try:
        # Ensure worker has access to global TOKENIZER_NAME if args doesn't contain it explicitly
        if not hasattr(args, 'tokenizer') or not args.tokenizer:
             # This case shouldn't happen if parse_args works correctly
             args.tokenizer = TOKENIZER_NAME

        return process_trajectory_worker(trajectory_data, trajectory_idx, session_start_time_ms, args)
    except Exception as e:
        # Log error within the wrapper, return empty list to avoid stopping map
        print(f"\n[Worker Wrapper Error] Trajectory {trajectory_idx}: {e}", file=sys.stderr)
        # import traceback # Uncomment for full traceback
        # traceback.print_exc()
        return [] # Important: return empty list on failure


# --- Main Processing Logic (Using executor.map) ---
def process_trajectories(args):
    print("--- Configuration ---")
    # (Print config as before)
    print(f"Dataset:         {args.hf_dataset} (Split: {args.hf_split})")
    print(f"Tokenizer:       {args.tokenizer}")
    print(f"Output File:     {args.output_file}")
    print(f"Workers:         {args.workers}")
    print(f"Block Size:      {args.block_size} tokens")
    print(f"Session Start Rate: {args.inter_session_start_rate:.2f} sess/s (Mean: {args.mean_inter_session_start_delay_ms:.2f} ms)")
    print(f"Intra-Session Rate: {args.inter_request_rate:.2f} req/s (Mean: {args.mean_inter_request_delay_ms:.2f} ms)")
    if args.num_trajectories:
        print(f"Processing Limit: {args.num_trajectories} trajectories")
    if args.cache_dir:
        print(f"Cache Dir:       {args.cache_dir}")
    print("---------------------")

    # --- Step 1: Scan dataset and Calculate Session Start Times ---
    print("\nStep 1: Scanning dataset and calculating session start times...")
    session_start_times = {}
    current_session_start_time_ms = 0.0
    trajectories_to_process_indices = []
    dataset_size_estimate = None # For progress bar if available

    try:
        # Try to get dataset size for better progress bar, fallback if streaming only
        try:
             # Load non-streaming first to get size if possible within reasonable time/memory
             # This might be slow/memory intensive for huge datasets, consider removing if problematic
             # dataset_info = load_dataset(args.hf_dataset, split=args.hf_split, streaming=False, cache_dir=args.cache_dir, download_mode='reuse_cache_if_exists')
             # dataset_size_estimate = len(dataset_info)
             # print(f"Estimated dataset size: {dataset_size_estimate}")
             # del dataset_info # Free memory
             pass # Skip size check for now to ensure low memory scan
        except Exception:
            print("Could not determine exact dataset size beforehand.")


        dataset_stream = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True, cache_dir=args.cache_dir)

        processed_count = 0
        scan_limit = args.num_trajectories # Use limit for tqdm total if set
        with tqdm(total=scan_limit, desc="Scanning Trajectories", unit="traj") as pbar:
            for i, trajectory_data in enumerate(dataset_stream):
                if scan_limit is not None and processed_count >= scan_limit:
                    if scan_limit is not None: pbar.total = processed_count # Adjust total if limit reached early
                    break

                # Basic validity check (can be enhanced)
                is_potentially_valid = (isinstance(trajectory_data, dict) and 'trajectory' in trajectory_data and isinstance(trajectory_data['trajectory'], list)) or \
                                       (isinstance(trajectory_data, list) and trajectory_data)

                if is_potentially_valid:
                    if processed_count > 0:
                        inter_session_delay_ms = generate_poisson_delay(args.inter_session_start_rate)
                        current_session_start_time_ms += inter_session_delay_ms

                    session_start_times[i] = current_session_start_time_ms
                    trajectories_to_process_indices.append(i)
                    processed_count += 1
                    pbar.update(1)

    except Exception as e:
        print(f"\nError during dataset scanning (Step 1): {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)

    if not trajectories_to_process_indices:
        print("\nNo valid trajectories found or selected. Exiting.", file=sys.stderr)
        return

    print(f"\nStep 1 Complete: Calculated start times for {len(trajectories_to_process_indices)} potential sessions.")


    # --- Step 2: Parallel Processing using executor.map ---
    print(f"\nStep 2: Processing {len(trajectories_to_process_indices)} trajectories using {args.workers} processes via map...")
    all_trace_entries = []
    map_args_list = [] # Prepare list of arguments for map

    try:
        print("Preparing arguments for parallel processing...")
        # Reload or rewind dataset stream
        dataset_process = load_dataset(args.hf_dataset, split=args.hf_split, streaming=True, cache_dir=args.cache_dir)

        prepared_count = 0
        with tqdm(total=len(trajectories_to_process_indices), desc="Preparing Map Args", unit="traj") as pbar_prep:
            for i, trajectory_data in enumerate(dataset_process):
                if i in session_start_times:
                    # Create tuple of arguments for the worker wrapper
                    arg_tuple = (
                        trajectory_data,
                        i,
                        session_start_times[i],
                        args # Pass the whole args object
                    )
                    map_args_list.append(arg_tuple)
                    prepared_count += 1
                    pbar_prep.update(1)

                # Stop iterating once all targeted trajectories are found
                if prepared_count >= len(trajectories_to_process_indices):
                    break # Found all needed data

        if len(map_args_list) != len(trajectories_to_process_indices):
             print(f"\nWarning: Prepared {len(map_args_list)} arguments, but expected {len(trajectories_to_process_indices)}. Proceeding with prepared args.", file=sys.stderr)

        if not map_args_list:
             print("\nNo arguments prepared for processing. Skipping Step 2.", file=sys.stderr)
        else:
            print(f"\nSubmitting {len(map_args_list)} tasks to ProcessPoolExecutor using map...")
            # Adjust chunksize dynamically? Or let map decide? Default often works well.
            # chunksize = max(1, len(map_args_list) // (args.workers * 2)) # Example dynamic chunksize
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
                 # Use executor.map with the wrapper function and the list of argument tuples
                 results_iterator = executor.map(worker_wrapper, map_args_list) # , chunksize=chunksize

                 # Iterate through results as they become available (map preserves order)
                 for result_entries in tqdm(results_iterator, total=len(map_args_list), desc="Processing Tasks (map)", unit="traj"):
                      if result_entries: # result_entries is the list returned by the worker
                           all_trace_entries.extend(result_entries)

    except Exception as e:
        print(f"\nError during parallel processing (Step 2 with map): {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


    print(f"\nStep 2 Complete: Collected {len(all_trace_entries)} trace entries.")

    if not all_trace_entries:
        print("No trace entries were generated after processing. Check worker logic or data. Exiting.")
        return

    # --- Step 3: Final Sort by Timestamp ---
    print("\nStep 3: Sorting all entries by timestamp...")
    all_trace_entries.sort(key=lambda x: x['timestamp'])
    print("Step 3 Complete.")

    # --- Step 4: Write Output ---
    output_path = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)

    print(f"\nStep 4: Writing {len(all_trace_entries)} trace entries to {output_path}")
    try:
        with open(output_path, 'w') as f:
            for entry in tqdm(all_trace_entries, desc="Writing Output", unit="req"):
                f.write(json.dumps(entry) + '\n')
    except IOError as e:
        print(f"\nError writing output file {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone!")


# --- Run the script ---
if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    process_trajectories(args)
    end_time = time.time()
    print(f"\nTotal processing time: {end_time - start_time:.2f} seconds")