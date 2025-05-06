#!/usr/bin/env python3
import json
import argparse

def compute_total_input_length(file_path, n=None):
    """
    Compute the total input_length from the first n requests in the dataset.
    
    Args:
        file_path (str): Path to the JSONL file
        n (int, optional): Number of requests to process. If None, process all requests.
    
    Returns:
        int: Total input length
    """
    total_length = 0
    count = 0
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if n is not None and count >= n:
                    break
                
                try:
                    request = json.loads(line)
                    total_length += request.get('input_length', 0)
                    count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {count + 1}. Skipping.")
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0
    
    print(f"Processed {count} requests.")
    return total_length

def main():
    parser = argparse.ArgumentParser(description='Compute total input length from a JSONL dataset.')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file')
    parser.add_argument('-n', type=int, default=None, 
                        help='Number of requests to process. If not specified, process all requests.')
    
    args = parser.parse_args()
    
    total_length = compute_total_input_length(args.file_path, args.n)
    print(f"Total input length: {total_length}")

if __name__ == "__main__":
    main()
