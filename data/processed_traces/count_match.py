import json
from collections import defaultdict

def calculate_prefix_cache_percentage(file_path, n):
    """
    Calculate the percentage of hash IDs that are part of a matching prefix cache.
    
    A prefix cache hit occurs when the first i hash IDs of the current request
    match the first i hash IDs of any previously seen request, where i is maximized.
    
    Args:
        file_path: Path to the JSONL file containing request data
        n: Number of entries to process
        
    Returns:
        Percentage of hash IDs that are part of a matching prefix
    """
    # Load and sort entries by timestamp
    entries = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:  # Stop after n entries
                break
            try:
                entry = json.loads(line.strip())
                # Ensure the entry has required fields
                if 'hash_ids' in entry and 'timestamp' in entry:
                    entries.append(entry)
                else:
                    print(f"Skipping entry at line {i+1}: missing required fields")
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i+1}")
                continue
    
    # Sort entries by timestamp
    entries.sort(key=lambda x: x['timestamp'])
    
    # Track previously seen hash ID sequences
    seen_hash_sequences = set()
    
    total_hash_ids = 0
    prefix_cache_hits = 0
    
    for entry in entries:
        hash_ids = entry.get('hash_ids', [])
        total_hash_ids += len(hash_ids)
        
        if not hash_ids:
            continue
        
        # Find the maximum prefix length that matches with any previous sequence
        max_prefix_length = 0
        hash_tuple = tuple(hash_ids)
        
        for prefix_len in range(1, len(hash_ids) + 1):
            current_prefix = hash_tuple[:prefix_len]
            if any(current_prefix == prev_seq[:prefix_len] for prev_seq in seen_hash_sequences if len(prev_seq) >= prefix_len):
                max_prefix_length = prefix_len
            else:
                break
        
        # Add to prefix cache hits
        prefix_cache_hits += max_prefix_length
        
        # Add current hash sequence to seen sequences
        seen_hash_sequences.add(hash_tuple)
    
    # Calculate percentage
    if total_hash_ids == 0:
        return 0.0
    
    percentage = (prefix_cache_hits / total_hash_ids) * 100
    return percentage

# Example usage
if __name__ == "__main__":
    file_path = './data/processed_traces/conversation_trace.jsonl'
    n = 23608  # Number of entries to process
    percentage = calculate_prefix_cache_percentage(file_path, n)
    print(f"Prefix cache percentage in first {n} entries: {percentage:.2f}%")