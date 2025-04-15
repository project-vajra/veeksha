import json
from collections import Counter

def count_repeated_hash_ids(file_path, n):
    hash_id_counts = Counter()
    total_hash_ids = 0
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:  # Stop after n entries
                break
            try:
                entry = json.loads(line.strip())
                hash_ids = entry.get('hash_ids', [])
                hash_id_counts.update(hash_ids)
                total_hash_ids += len(hash_ids)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON at line {i+1}")
                continue
    
    # Count hash_ids that appear more than once
    repeated_count = sum(count for count in hash_id_counts.values() if count > 1)
    
    # Calculate percentage
    if total_hash_ids == 0:
        return 0.0
    percentage = (repeated_count / total_hash_ids) * 100
    
    return percentage

# Example usage
file_path = './data/processed_traces/conversation_trace.jsonl'
n = 100000  # Number of entries to process
percentage = count_repeated_hash_ids(file_path, n)
print(f"Percentage of repeated hash_ids in first {n} entries: {percentage:.2f}%")