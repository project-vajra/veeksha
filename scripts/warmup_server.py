import requests
import sys
import time
import json

def warmup(base_url="http://localhost:30002"):
    print(f"Connecting to {base_url}...")
    
    # 1. Get Models
    try:
        response = requests.get(f"{base_url}/v1/models")
        response.raise_for_status()
        models = response.json()
        # Handle both list formats commonly seen
        if isinstance(models, dict) and 'data' in models:
            model_list = models['data']
        elif isinstance(models, list):
            model_list = models
        else:
            print(f"Unexpected /models response format: {models}")
            model_list = []

        if model_list:
            # Just pick the first one's ID
            model_id = model_list[0]['id']
            print(f"Found model: {model_id}")
        else:
            print("No models returned by API.")
            model_id = "default"
            
    except Exception as e:
        print(f"Error fetching models (might be offline or different API): {e}")
        print("Using default model 'default'")
        model_id = "default"

    # 2. Send Chat Completion
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hello! This is a warmup request."}
        ],
        "max_tokens": 20
    }
    
    print(f"Sending warmup request to {base_url}/v1/chat/completions with model '{model_id}'...")
    start_time = time.time()
    
    response = None
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        print("Response received!")
        print(f"Time taken: {time.time() - start_time:.2f}s")
        try:
            content = response.json()['choices'][0]['message']['content']
            print(f"Content: {content}")
        except (KeyError, IndexError, json.JSONDecodeError):
             print(f"Raw Response: {response.text}")
             
    except Exception as e:
        print(f"Error sending request: {e}")
        if response is not None:
             print(f"Status Code: {response.status_code}")
             print(f"Response Text: {response.text}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:30003"
    warmup(url)
