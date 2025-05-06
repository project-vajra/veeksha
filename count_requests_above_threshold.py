#!/usr/bin/env python3
import json
import argparse

def count_requests_above_threshold(file_path, threshold, n=None):
    """
    Count the number of requests with input_length >= threshold from the dataset.
    
    Args:
        file_path (str): Path to the JSONL file
        threshold (int): Minimum input length threshold
        n (int, optional): Number of requests to process. If None, process all requests.
    
    Returns:
        tuple: (count of requests with input_length >= threshold, id of first request meeting threshold)
    """
    count_above_threshold = 0
    total_processed = 0
    first_request_id = None
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if n is not None and total_processed >= n:
                    break
                
                try:
                    request = json.loads(line)
                    input_length = request.get('input_length', 0)
                    
                    if input_length >= threshold:
                        count_above_threshold += 1
                        
                        # Store the ID of the first request that meets the threshold
                        if first_request_id is None:
                            # Try to get request_id, or fall back to other identifiers
                            first_request_id = request.get('request_id')
                            if first_request_id is None:
                                # Try other possible ID fields
                                first_request_id = request.get('id', request.get('trajectory_id', 'unknown'))
                    
                    total_processed += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {total_processed + 1}. Skipping.")
                    continue
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0, None
    
    print(f"Processed {total_processed} requests.")
    return count_above_threshold, first_request_id

def main():
    parser = argparse.ArgumentParser(description='Count requests with input length above a threshold.')
    parser.add_argument('file_path', type=str, help='Path to the JSONL file')
    parser.add_argument('threshold', type=int, help='Minimum input length threshold')
    parser.add_argument('-n', type=int, default=None, 
                        help='Number of requests to process. If not specified, process all requests.')
    
    args = parser.parse_args()
    
    count, first_id = count_requests_above_threshold(args.file_path, args.threshold, args.n)
    print(f"Number of requests with input length >= {args.threshold}: {count}")
    if first_id is not None:
        print(f"ID of first request with input length >= {args.threshold}: {first_id}")
    else:
        print(f"No requests found with input length >= {args.threshold}")

if __name__ == "__main__":
    main()
