#%%
import copy
import json
import math
import random
import pandas as pd
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import sys

random.seed(42)

#%%
# --- Command Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Generate traces with different session sampled with different dispatch rate')
    parser.add_argument('--trace-file', help='Path to the trace file')
    parser.add_argument('--block-size', type=int, default=512, help='Block size in tokens')
    parser.add_argument('--output-dir', default='./data/generated_traces', help='Directory to save the report and plots')
    parser.add_argument('--dispatch-rate', type=int, default=1, help='Dispatch rate')
    parser.add_argument('--minimum-match-threshold', type=float, default=0.1, help='Minimum match threshold')
    args = parser.parse_args()

    return args

#%%
def load_trace_data(trace_file):
    """Load and parse trace data from a file."""
    data = []
    try:
        with open(trace_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {trace_file}")
    except FileNotFoundError:
        print(f"Error: Trace file '{trace_file}' not found.")
        return None

    if not data:
        print(f"No valid data loaded from {trace_file}.")
        return None
    
    # Convert to DataFrame and sort by timestamp
    df = pd.DataFrame(data)
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)  # Reset index after sorting
    
    return df

#%%
def create_running_hashes(hash_ids):
    """Create running hashes for a sequence of hash IDs.
    Each running hash at position i is a hash of the sequence from 0 to i."""
    if not hash_ids:
        return []
    
    running_hashes = []
    
    # For each position i, create a hash of the tuple of all elements from 0 to i
    for i in range(len(hash_ids)):
        # Create a tuple of the prefix and hash it
        prefix_tuple = tuple(hash_ids[:i+1])
        prefix_hash = hash(prefix_tuple) % (2**32)  # Keep hash size manageable
        running_hashes.append(prefix_hash)
    
    return running_hashes

#%%
def get_sessions(requests, minimum_match_threshold=0.1):
    request_to_session = {}
    hash_to_request = {}
    hash_to_length = {}
    sessions = {}

    # insert all the hash_ids into the hash_to_request
    for idx, request in enumerate(requests):
        hash_ids = request['hash_ids']
        for hash_id in hash_ids:
            if hash_id not in hash_to_request:
                hash_to_request[hash_id] = idx

    # Process each request in order
    for idx, request in enumerate(requests):
        current_hashes = request['hash_ids']
        best_match_hash = None
        best_match_length = 0
        
        # Check for matches of increasing length
        for prefix_len, hash_id in enumerate(current_hashes):
            # If this prefix exists in our map, we have a match
            if hash_id in hash_to_length:
                best_match_hash = hash_id
                best_match_length = prefix_len + 1

        if best_match_length > minimum_match_threshold * len(current_hashes):
            # match to the existing session
            
            matched_request_idx = hash_to_request[best_match_hash]
            session_id = request_to_session[matched_request_idx]
            request_to_session[idx] = session_id
            request['session_id'] = session_id
            sessions[session_id].append(request)
        else:
            # create a new session
            request_to_session[idx] = idx
            request['session_id'] = idx
            sessions[idx] = [request]

        # Add all prefixes of the current hash sequence to our map
        for prefix_len, hash_id in enumerate(current_hashes):
            hash_to_length[hash_id] = prefix_len + 1

    return sessions

#%%
def sample_sessions(sessions, dispatch_rate):
    """Sample sessions using dispatch rate with poisson distribution."""
    sampled_sessions = []

    sessions = list(sessions.values())
    # shuffle the sessions
    random.shuffle(sessions)

    timestamp = 0

    while sessions:
        next_interval = -math.log(1.0 - random.random()) / dispatch_rate
        next_interval = min(next_interval, 1 / dispatch_rate * 3) * 1000
        session = sessions.pop(0)
        session_original_timestamp = None
        for request in session:
            if session_original_timestamp is None:
                session_original_timestamp = request['timestamp']
            request['timestamp'] = timestamp + (request['timestamp'] - session_original_timestamp)
        sampled_sessions.append(session)
        timestamp += next_interval

    return sampled_sessions


#%%
def analyze_single_trace(df, args):
    """Analyze a single trace file and generate a report."""
    # Store the original hash IDs before replacing them
    df['original_hash_ids'] = df['hash_ids']

    # Replace hash_ids with running hashes
    df['hash_ids'] = df.apply(lambda row: create_running_hashes(row['original_hash_ids']), axis=1)

    # Verify the transformation
    if len(df) > 0:
        print("Created running hashes for {0} requests".format(len(df)))
    print()

    # convert the df into list of dict
    requests = df.to_dict('records')

    # 1. Calculate prefix match using running hashes and a hash map for efficient lookups
    print("Analyzing prefix matches...")
    
    # 2. Get sessions
    sessions = get_sessions(requests, args.minimum_match_threshold)

    print("Created {0} sessions for {1} requests".format(len(sessions), len(requests)))


    # print the stats of the session length min, max, std, mean, median, p25, p75, p90
    session_lengths = [len(session) for session in sessions.values()]
    print(f"Session max length: {max(session_lengths)}")
    print(f"Session min length: {min(session_lengths)}")
    print(f"Session mean length: {np.mean(session_lengths)}")
    print(f"Session std length: {np.std(session_lengths)}")
    print(f"Session median length: {np.median(session_lengths)}")
    print(f"Session p25 length: {np.percentile(session_lengths, 25)}")
    print(f"Session p75 length: {np.percentile(session_lengths, 75)}")
    print(f"Session p90 length: {np.percentile(session_lengths, 90)}")
    print()

    # 3. Sample sessions using dispatch rate with poisson distribution
    sampled_sessions = sample_sessions(sessions, args.dispatch_rate)


    # 4. flatten the sessions
    sampled_requests = [request for session in sampled_sessions for request in session]

    # sort by timestamp
    sampled_requests.sort(key=lambda x: x['timestamp'])

    return sampled_requests, sampled_sessions



#%%
def main():
    args = parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Create a specific directory for this trace and timestamp
    trace_dir = os.path.join(args.output_dir, args.trace_file, str(args.dispatch_rate))
    os.makedirs(trace_dir, exist_ok=True)
        
    # Load trace data
    df = load_trace_data(args.trace_file)
    if df is None:
        print("Error: Could not load trace file {0}".format(args.trace_file))
        return

    # Analyze the trace
    sampled_requests, sampled_sessions = analyze_single_trace(df, args)

    # Save the sampled trace as a jsonl
    with open(os.path.join(trace_dir, f'sampled_trace_dr{args.dispatch_rate}_mmt{args.minimum_match_threshold}.jsonl'), 'w') as f:
        for request in sampled_requests:
            json.dump(request, f)
            f.write('\n')

    # save the sampled sessions in individual files
    sessions_dir = os.path.join(trace_dir, 'sessions')
    os.makedirs(sessions_dir, exist_ok=True)
    
    for i, session in enumerate(sampled_sessions[:100]):
        with open(os.path.join(sessions_dir, f'sampled_session_{i}.jsonl'), 'w') as f:
            for request in session:
                json.dump(request, f)
                f.write('\n')


if __name__ == "__main__":
    main()

# %%
