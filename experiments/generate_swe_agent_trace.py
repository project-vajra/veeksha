import json
import time
import random
import math
import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse
import os
import sys
import concurrent.futures
from functools import lru_cache
from transformers import AutoTokenizer

# --- Configuration ---
HF_DATASET_NAME = "nebius/SWE-agent-trajectories"
HF_DATASET_SPLIT = "train" # Or 'dev', 'test'
OUTPUT_FILENAME = "swe_agent_trace.jsonl"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"  # Llama tokenizer

# Timestamp Simulation Parameters
# Mean time between sessions (in milliseconds)
MEAN_INTER_SESSION_TIME_MS = 5000  # 5 seconds between sessions

# Mean time between requests within a session (in milliseconds)
MEAN_INTER_REQUEST_TIME_MS = 500   # 0.5 seconds between requests in a session

# Block size for chunking text (in tokens)
BLOCK_SIZE = 512

# Number of worker threads
NUM_WORKERS = 64

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description='Convert SWE-Agent trajectories to trace format with Poisson-distributed delays.')
    parser.add_argument('--output-file', default=OUTPUT_FILENAME, help=f'Output filename for the converted trace (default: {OUTPUT_FILENAME})')
    parser.add_argument('--hf-dataset', default=HF_DATASET_NAME, help=f'Hugging Face dataset name (default: {HF_DATASET_NAME})')
    parser.add_argument('--hf-split', default=HF_DATASET_SPLIT, help=f'Dataset split to use (default: {HF_DATASET_SPLIT})')
    parser.add_argument('--tokenizer', default=TOKENIZER_NAME, help=f'Tokenizer to use for token counting (default: {TOKENIZER_NAME})')
    parser.add_argument('--block-size', type=int, default=BLOCK_SIZE, help=f'Block size for chunking text (default: {BLOCK_SIZE} tokens)')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS, help=f'Number of worker threads (default: {NUM_WORKERS})')
    parser.add_argument('--inter-session-rate', type=float, default=1000/MEAN_INTER_SESSION_TIME_MS, 
                        help=f'Rate parameter for Poisson distribution of inter-session delays (default: {1000/MEAN_INTER_SESSION_TIME_MS} req/sec)')
    parser.add_argument('--inter-request-rate', type=float, default=1000/MEAN_INTER_REQUEST_TIME_MS, 
                        help=f'Rate parameter for Poisson distribution of inter-request delays within a session (default: {1000/MEAN_INTER_REQUEST_TIME_MS} req/sec)')
    return parser.parse_args()

# --- Helper Functions ---
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

def generate_poisson_delay(rate):
    """Generate a delay from a Poisson process with given rate parameter.
    
    Args:
        rate: Rate parameter in events per second
        
    Returns:
        Delay in milliseconds
    """
    # Generate exponential inter-arrival time
    delay_sec = -math.log(1.0 - random.random()) / rate
    
    # Cap the delay to avoid extreme outliers (3x the mean)
    mean_delay_sec = 1.0 / rate
    delay_sec = min(delay_sec, mean_delay_sec * 3)
    
    # Convert to milliseconds
    return delay_sec * 1000

# Create a cached tokenize function to avoid repeated tokenization
def create_cached_tokenizer(tokenizer):
    @lru_cache(maxsize=10000)  # Cache up to 10,000 tokenizations
    def cached_tokenize(text):
        return tokenizer.encode(text)
    
    return cached_tokenize

def chunk_and_hash_text(token_ids, block_size):
    """Chunk tokenized text into blocks of specified token size and hash each block.
    
    Args:
        token_ids: List of token IDs
        block_size: Size of each block in tokens
        
    Returns:
        List of hash IDs for each block
    """
    # Chunk the tokens into blocks of block_size
    hash_ids = []
    for i in range(0, len(token_ids), block_size):
        # Get the block of tokens
        block = token_ids[i:i+block_size]
        
        # Hash the block
        block_hash = hash(tuple(block)) % (2**32)
        hash_ids.append(block_hash)
    
    return hash_ids

# --- Process a single trajectory ---
def process_single_trajectory(trajectory_data, tokenize_fn, args, trajectory_idx, start_timestamp=0):
    """Process a single trajectory and return trace entries."""
    # Print the first item to debug
    if trajectory_idx == 0:
        print(f"Sample trajectory data: {str(trajectory_data)[:500]}...")
    
    # Extract the trajectory
    if isinstance(trajectory_data, dict) and 'trajectory' in trajectory_data:
        trajectory = trajectory_data['trajectory']
    else:
        # The dataset might directly provide the trajectory list
        trajectory = trajectory_data
    
    # Ensure trajectory is a list
    if not isinstance(trajectory, list):
        print(f"Warning: Trajectory {trajectory_idx} is not a list, skipping")
        return [], start_timestamp
    
    if not trajectory:
        print(f"Warning: Trajectory {trajectory_idx} is empty, skipping")
        return [], start_timestamp
    
    # Initialize for this trajectory
    trace_entries = []
    current_timestamp_ms = start_timestamp
    
    # Add inter-session delay (except for the first trajectory)
    if trajectory_idx > 0:
        current_timestamp_ms += generate_poisson_delay(args.inter_session_rate)
    
    # Process each turn in the trajectory
    conversation_history = ""
    current_input = ""
    current_output = ""
    
    # Cache for tokenized texts to avoid repeated tokenization
    token_cache = {}
    
    for turn_idx, turn in enumerate(trajectory):
        # Make sure turn is a dictionary
        if not isinstance(turn, dict):
            print(f"Warning: Turn {turn_idx} in trajectory {trajectory_idx} is not a dictionary, skipping")
            continue
        
        role = turn.get('role', '')
        text = turn.get('text', '')
        
        # Skip empty turns
        if not text:
            continue
        
        # Handle different roles
        if role == 'system':
            # System prompts are part of the initial context
            conversation_history += f"System: {text}\n\n"
        
        elif role == 'user':
            # User messages become part of the input
            current_input = f"User: {text}\n\n"
            
            # The full input is the conversation history plus the current input
            full_input = conversation_history + current_input
            
            # Tokenize the full input (using cache)
            if full_input not in token_cache:
                token_cache[full_input] = tokenize_fn(full_input)
            input_tokens = token_cache[full_input]
            
            # If this is not the first user message, it's a new request
            if turn_idx > 1:
                # Add inter-request delay
                current_timestamp_ms += generate_poisson_delay(args.inter_request_rate)
                
                # Generate hash IDs by chunking the tokenized input
                hash_ids = chunk_and_hash_text(input_tokens, args.block_size)
                
                # Tokenize the output (using cache)
                if current_output not in token_cache and current_output:
                    token_cache[current_output] = tokenize_fn(current_output)
                output_tokens = token_cache.get(current_output, []) if current_output else []
                
                # Create trace entry
                trace_entry = {
                    'timestamp': current_timestamp_ms,
                    'hash_ids': hash_ids,
                    'input_length': len(input_tokens),
                    'output_length': len(output_tokens),
                    'trajectory_id': trajectory_idx,
                    'turn_id': turn_idx
                }
                
                trace_entries.append(trace_entry)
        
        elif role == 'ai' or role == 'assistant':
            # AI/assistant messages become the output
            current_output = f"Assistant: {text}\n\n"
            
            # If this is the first AI message and we haven't created a trace entry yet
            if turn_idx == 1 or (turn_idx > 1 and not any(entry['turn_id'] == turn_idx-1 for entry in trace_entries)):
                # The full input is the conversation history
                full_input = conversation_history
                
                # Tokenize the input (using cache)
                if full_input not in token_cache:
                    token_cache[full_input] = tokenize_fn(full_input)
                input_tokens = token_cache[full_input]
                
                # Generate hash IDs by chunking the tokenized input
                hash_ids = chunk_and_hash_text(input_tokens, args.block_size)
                
                # Tokenize the output (using cache)
                if current_output not in token_cache:
                    token_cache[current_output] = tokenize_fn(current_output)
                output_tokens = token_cache[current_output]
                
                # Create trace entry for the first interaction
                trace_entry = {
                    'timestamp': current_timestamp_ms,
                    'hash_ids': hash_ids,
                    'input_length': len(input_tokens),
                    'output_length': len(output_tokens),
                    'trajectory_id': trajectory_idx,
                    'turn_id': turn_idx
                }
                
                trace_entries.append(trace_entry)
            
            # Add the current exchange to the conversation history
            conversation_history += current_input + current_output
            current_input = ""
            current_output = ""
    
    return trace_entries, current_timestamp_ms

# --- Main Processing Logic ---
def process_trajectories(args):
    """Process SWE-Agent trajectories and convert to trace format with realistic delays."""
    print(f"Loading dataset: {args.hf_dataset}, split: {args.hf_split}")
    dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    
    # Examine the dataset structure
    print(f"Dataset has {len(dataset)} items")
    print(f"Dataset features: {dataset.features}")
    
    # Load tokenizer for accurate token counting
    print(f"Loading tokenizer: {args.tokenizer}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        # Create a cached tokenize function
        tokenize_fn = create_cached_tokenizer(tokenizer)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Cannot proceed without tokenizer")
        return
    
    # Process a single trajectory first to understand the structure
    print("Processing a single trajectory to understand the structure...")
    first_entries, _ = process_single_trajectory(dataset[0], tokenize_fn, args, 0, 0)
    if not first_entries:
        print("Failed to process the first trajectory. Please check the dataset structure.")
        return
    
    print(f"Successfully processed first trajectory with {len(first_entries)} entries")
    
    # Split dataset into chunks for parallel processing
    num_trajectories = len(dataset)
    chunk_size = max(1, num_trajectories // args.workers)
    trajectory_chunks = []
    
    for i in range(0, num_trajectories, chunk_size):
        end_idx = min(i + chunk_size, num_trajectories)
        chunk = [dataset[j] for j in range(i, end_idx)]
        trajectory_chunks.append(chunk)
    
    print(f"Processing {num_trajectories} trajectories using {args.workers} workers")
    print(f"Split into {len(trajectory_chunks)} chunks of approximately {chunk_size} trajectories each")
    
    # Process trajectories in parallel
    all_trace_entries = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit tasks for each chunk
        futures = []
        for chunk_idx, chunk in enumerate(trajectory_chunks):
            # Calculate starting trajectory index for this chunk
            start_idx = chunk_idx * chunk_size
            
            # Submit the task
            future = executor.submit(
                process_chunk, 
                chunk, 
                tokenize_fn, 
                args, 
                start_idx
            )
            futures.append(future)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                chunk_entries = future.result()
                all_trace_entries.extend(chunk_entries)
                print(f"Processed chunk with {len(chunk_entries)} entries")
            except Exception as e:
                print(f"Error processing chunk: {e}")
                import traceback
                traceback.print_exc()
    
    # Sort all entries by timestamp
    all_trace_entries.sort(key=lambda x: x['timestamp'])
    
    # Write trace to output file
    output_path = args.output_file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Writing {len(all_trace_entries)} trace entries to {output_path}")
    with open(output_path, 'w') as f:
        for entry in all_trace_entries:
            f.write(json.dumps(entry) + '\n')
    
    print("Done!")

def process_chunk(chunk, tokenize_fn, args, start_idx):
    """Process a chunk of trajectories and return trace entries."""
    chunk_entries = []
    current_timestamp = 0
    
    for i, trajectory_data in enumerate(chunk):
        trajectory_idx = start_idx + i
        try:
            entries, current_timestamp = process_single_trajectory(
                trajectory_data, 
                tokenize_fn, 
                args, 
                trajectory_idx, 
                current_timestamp
            )
            chunk_entries.extend(entries)
            if i % 10 == 0:
                print(f"Processed {i}/{len(chunk)} trajectories in chunk starting at {start_idx}")
        except Exception as e:
            print(f"Error processing trajectory {trajectory_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    return chunk_entries

# --- Run the script ---
if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    process_trajectories(args)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")