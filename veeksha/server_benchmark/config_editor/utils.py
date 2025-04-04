from jinja2 import Environment
import yaml
import os
import csv
import time

def load_yaml_with_jinja_expressions(filepath):
    with open(filepath, 'r') as f:
        raw_yaml = f.read()
    
    # First pass to extract variables
    first_pass = yaml.safe_load(raw_yaml)
    
    # Create a template with the YAML content
    env = Environment()
    # Add a filter for mathematical expressions
    env.filters['calc'] = lambda x: eval(x, {"__builtins__": {}}, {})
    template = env.from_string(raw_yaml)
    
    # Render the template with the variables from the first pass
    rendered = template.render(**first_pass)
    
    # Parse the rendered YAML
    return yaml.safe_load(rendered)


def generate_trace_csv(spec_type, output_path=None):
    """
    Generate a CSV trace file based on the specified template spec type.
    
    Args:
        spec_type (str): The type of spec to use ('prefill' or 'decode')
        output_path (str, optional): Path where the CSV file should be saved.
            If None, a default path will be used.
            
    Returns:
        str: Path to the generated CSV file
    """
    if spec_type not in ['prefill', 'decode']:
        raise ValueError(f"Unknown spec type: {spec_type}. Must be 'prefill' or 'decode'")
    
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    specs_dir = os.path.join(base_dir, "trace_file_specs")
    
    # Generate trace data and extract key parameters based on the spec type
    if spec_type == 'prefill':
        # For prefill experiments, we use hardcoded values based on the template
        # Read the template file to extract non-Jinja values
        try:
            with open(os.path.join(specs_dir, "prefill_experiment_spec.jinja"), 'r') as f:
                lines = f.readlines()
            
            # Extract values using simple parsing (avoiding YAML parser issues with Jinja)
            spec = {}
            for line in lines:
                if ':' in line and '{{' not in line:  # Skip lines with Jinja templates
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    # Extract the value, removing comments
                    value_part = parts[1].strip()
                    if '#' in value_part:
                        value_part = value_part.split('#', 1)[0].strip()
                    try:
                        # Try to convert to int if possible
                        value = int(value_part)
                    except ValueError:
                        value = value_part
                    spec[key] = value
            
            # Use default values for any missing keys
            prefill_spec = {
                'batch_size': spec.get('batch_size', 1),
                'num_requests': spec.get('num_requests', 100),
                'prefill_size': spec.get('prefill_size', 4000),
                'decode_size': spec.get('decode_size', 1)
            }
            
            # Create a filename based on key parameters
            filename_parts = [
                f"prefill{prefill_spec['prefill_size']}",
                f"decode{prefill_spec['decode_size']}",
                f"req{prefill_spec['num_requests']}"
            ]
            filename_base = "_".join(filename_parts)
            
            csv_data = generate_prefill_trace(prefill_spec)
        except Exception as e:
            print(f"Warning: Error parsing prefill spec file: {e}")
            # Fallback to default values
            prefill_spec = {
                'batch_size': 1,
                'num_requests': 100,
                'prefill_size': 4000,
                'decode_size': 1
            }
            filename_base = "prefill4000_decode1_req100"
            csv_data = generate_prefill_trace(prefill_spec)
    else:  # decode
        # For decode experiments, we use hardcoded values based on the template
        try:
            with open(os.path.join(specs_dir, "decode_experiment_spec.jinja"), 'r') as f:
                lines = f.readlines()
            
            # Extract values using simple parsing (avoiding YAML parser issues with Jinja)
            spec = {}
            for line in lines:
                if ':' in line and '{{' not in line:  # Skip lines with Jinja templates
                    parts = line.split(':', 1)
                    key = parts[0].strip()
                    # Extract the value, removing comments
                    value_part = parts[1].strip()
                    if '#' in value_part:
                        value_part = value_part.split('#', 1)[0].strip()
                    try:
                        # Try to convert to int if possible
                        value = int(value_part)
                    except ValueError:
                        value = value_part
                    spec[key] = value
            
            # Use the extracted values and calculate the derived values
            batch_size = spec.get('batch_size', 64)
            profiling_iterations = spec.get('profiling_iterations', 100)
            
            decode_spec = {
                'batch_size': batch_size,
                'profiling_iterations': profiling_iterations,
                'num_requests': batch_size,
                'prefill_tokens': batch_size,
                'decode_tokens': batch_size + profiling_iterations
            }
            
            # Create a filename based on key parameters
            filename_parts = [
                f"batch{decode_spec['batch_size']}",
                f"prefill{decode_spec['prefill_tokens']}",
                f"decode{decode_spec['decode_tokens']}",
                f"prof{decode_spec['profiling_iterations']}"
            ]
            filename_base = "_".join(filename_parts)
            
            csv_data = generate_decode_trace(decode_spec)
        except Exception as e:
            print(f"Warning: Error parsing decode spec file: {e}")
            # Fallback to default values
            decode_spec = {
                'batch_size': 64,
                'profiling_iterations': 100,
                'num_requests': 64,
                'prefill_tokens': 64,
                'decode_tokens': 164
            }
            filename_base = "batch64_prefill64_decode164_prof100"
            csv_data = generate_decode_trace(decode_spec)
    
    # Set default output path if not provided
    if output_path is None:
        # Create the output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(base_dir))), "data", "generated_traces")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a filename that includes the template values without timestamp
        output_path = os.path.join(output_dir, f"{filename_base}.csv")
        
        # Check if file already exists and handle it
        if os.path.exists(output_path):
            # If the file already exists, we'll overwrite it
            # This ensures we always have the latest version of the trace
            print(f"Overwriting existing trace file: {output_path}")
    
    # Write the CSV file
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['num_prefill_tokens', 'num_decode_tokens', 'num_total_tokens', 'pd_ratio'])
        for row in csv_data:
            writer.writerow(row)
    
    return output_path

def generate_prefill_trace(spec):
    """
    Generate trace data for prefill experiments based on the spec.
    
    Args:
        spec (dict): The specification loaded from the prefill template
        
    Returns:
        list: List of rows for the CSV file, each containing 
              [num_prefill_tokens, num_decode_tokens, num_total_tokens, pd_ratio]
    """
    batch_size = spec.get('batch_size', 1)
    num_requests = spec.get('num_requests', 100)
    prefill_size = spec.get('prefill_size', 4000)
    decode_size = spec.get('decode_size', 1)
    
    # Generate the trace data
    trace_data = []
    
    for _ in range(num_requests):
        # For prefill experiments, we keep the prefill size constant
        # and the decode size small to focus on prefill performance
        num_prefill = prefill_size
        num_decode = decode_size
        num_total = num_prefill + num_decode
        pd_ratio = num_prefill / num_decode if num_decode > 0 else num_prefill
        
        trace_data.append([num_prefill, num_decode, num_total, pd_ratio])
    
    return trace_data

def generate_decode_trace(spec):
    """
    Generate trace data for decode experiments based on the spec.
    
    Args:
        spec (dict): The specification loaded from the decode template
        
    Returns:
        list: List of rows for the CSV file, each containing 
              [num_prefill_tokens, num_decode_tokens, num_total_tokens, pd_ratio]
    """
    batch_size = spec.get('batch_size', 64)
    profiling_iterations = spec.get('profiling_iterations', 100)
    num_requests = spec.get('num_requests', batch_size)
    prefill_tokens = spec.get('prefill_tokens', batch_size)
    decode_tokens = spec.get('decode_tokens', batch_size + profiling_iterations)
    
    # Generate the trace data
    trace_data = []
    
    for _ in range(num_requests):
        # For decode experiments, we keep the prefill size small
        # and the decode size large to focus on decode performance
        num_prefill = prefill_tokens
        num_decode = decode_tokens
        num_total = num_prefill + num_decode
        pd_ratio = num_prefill / num_decode if num_decode > 0 else num_prefill
        
        trace_data.append([num_prefill, num_decode, num_total, pd_ratio])
    
    return trace_data
