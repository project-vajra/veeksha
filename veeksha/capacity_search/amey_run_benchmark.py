import subprocess
import time
import os
import signal
import shlex
import argparse
import itertools
import yaml
import requests
import re # For model family extraction AND error pattern matching
from pathlib import Path # For easier path manipulation
from requests.exceptions import ConnectionError, Timeout
from typing import List, Tuple, Dict, Optional, Any, IO
import hashlib
import json
import datetime
import traceback # For printing stack traces

# --- Constants ---
COMPLETION_MARKER_FILENAME = ".benchmark_complete"

# --- Helper Functions (setup_logging, run_command, kill_process_group_and_close_logs) ---
# ... (setup_logging, run_command, kill_process_group_and_close_logs remain the same as Version 3) ...
# (Include the setup_logging, run_command, kill_process_group_and_close_logs from the previous version)
def setup_logging(log_dir: str, name_prefix: str) -> Tuple[str, str]:
    """Creates log directory and returns paths for stdout and stderr log files."""
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, f"{name_prefix}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{name_prefix}_stderr.log")
    return stdout_log_path, stderr_log_path

def run_command(
    cmd_list: List[str],
    env_name: Optional[str] = None,
    conda_base_env: Optional[str] = None, # Added for flexibility
    popen: bool = False,
    check: bool = True, # Still useful for raising errors when needed
    stdout_log_path: Optional[str] = None,
    stderr_log_path: Optional[str] = None,
    stream_logs: bool = False,
) -> Any: # Return type depends on popen
    """
    Runs a command, optionally within a conda environment, prints it,
    handles execution, manages logging, and returns process or result.

    Returns:
        - If popen=True: Tuple[subprocess.Popen, Optional[IO[Any]], Optional[IO[Any]]]
        - If popen=False: subprocess.CompletedProcess object
    """
    env_vars = os.environ.copy()
    full_cmd: List[str] = [] # Initialize

    if env_name:
        # Determine conda base path
        conda_base = conda_base_env or os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_ROOT")
        if not conda_base and env_name:
            try:
                conda_info_cmd = ["conda", "info", "--base"]
                result = subprocess.run(conda_info_cmd, capture_output=True, text=True, check=True, timeout=5)
                conda_base = result.stdout.strip()
                print(f"  Auto-detected conda base: {conda_base}", flush=True)
            except Exception: # Catch broader exceptions including timeout
                # Fallback heuristic - adjust if needed
                home = os.path.expanduser("~")
                potential_bases = [f"{home}/miniconda3", f"{home}/anaconda3", f"{home}/miniforge3"]
                found_base = None
                for pb in potential_bases:
                    if os.path.exists(os.path.join(pb, "envs", env_name)):
                        found_base = pb
                        break
                if found_base:
                    conda_base = found_base
                    print(f"  Warning: Cannot auto-detect conda base. Using guessed path: {conda_base}", flush=True)
                else:
                    conda_base = None # Indicate failure
                    print(f"  Error: Cannot auto-detect or guess conda base. Set 'conda_base_env' in config or ensure conda is in PATH.", flush=True)


        conda_env_path = None
        if conda_base:
            conda_env_path = Path(conda_base) / "envs" / env_name
            lib_path = conda_env_path / "lib"
            if lib_path.is_dir():
                existing_ld_path = env_vars.get("LD_LIBRARY_PATH", "")
                env_vars["LD_LIBRARY_PATH"] = f"{lib_path}:{existing_ld_path}" if existing_ld_path else str(lib_path)
            # else: # Reduce noise
            #     print(f"  Note: Conda env lib path not found: {lib_path}", flush=True)

        # Prefer using prefix if the path exists
        if conda_env_path and conda_env_path.exists():
             full_cmd = ["conda", "run", "--no-capture-output", "--prefix", str(conda_env_path)] + cmd_list
        elif env_name and conda_base: # Fallback to name if path check fails but base found
             print(f"  Warning: Conda env path for '{env_name}' not found, attempting activation by name.", flush=True)
             full_cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd_list
        elif env_name: # Fallback if no base was found/guessed
             print(f"  Error: Cannot activate conda env '{env_name}' due to missing base path. Command will run in current env.", flush=True)
             full_cmd = cmd_list # Run directly if env specified but unusable
        else: # No env name provided
            full_cmd = cmd_list
    else:
        full_cmd = cmd_list

    cmd_str = ' '.join(shlex.quote(str(part)) for part in full_cmd)
    print(f"\nExecuting{' in env ' + shlex.quote(env_name) if env_name else ''}: {cmd_str}", flush=True)
    if stdout_log_path:
        print(f"  stdout log: {stdout_log_path}", flush=True)
    if stderr_log_path:
        print(f"  stderr log: {stderr_log_path}", flush=True)

    stdout_f, stderr_f = None, None
    try:
        if stdout_log_path:
            stdout_f = open(stdout_log_path, 'w')
        if stderr_log_path:
            stderr_f = open(stderr_log_path, 'w')

        if popen:
            process = subprocess.Popen(
                full_cmd,
                preexec_fn=os.setsid,
                stdout=stdout_f,
                stderr=stderr_f,
                env=env_vars
            )
            # Return tuple immediately for popen
            return process, stdout_f, stderr_f
        else:
            # Run synchronously and capture output
            result = subprocess.run(
                full_cmd,
                check=False, # Don't raise exception immediately, return result
                text=True,
                capture_output=True,
                env=env_vars
            )
             # Write captured output AFTER completion
            if stdout_f and result.stdout: stdout_f.write(result.stdout); stdout_f.flush()
            if stderr_f and result.stderr: stderr_f.write(result.stderr); stderr_f.flush()

            print(f"Command finished with exit code {result.returncode}.")
            if stream_logs:
                 # Stream AFTER writing to file
                print("\n--- Benchmark STDOUT ---")
                print(result.stdout or "<No stdout>")
                print("--- Benchmark STDERR ---")
                print(result.stderr or "<No stderr>")
                print("--- End Logs ---\n")
            else:
                 # Show snippets even if not fully streaming
                 stdout_suffix = "..." if len(result.stdout or "") > 500 else ""
                 stderr_suffix = "..." if len(result.stderr or "") > 500 else ""
                 print(f"STDOUT (last 500 chars):\n{(result.stdout or '')[-500:]}{stdout_suffix}")
                 print(f"STDERR (last 500 chars):\n{(result.stderr or '')[-500:]}{stderr_suffix}")

            if check and result.returncode != 0:
                # Raise error only if check=True and failed
                raise subprocess.CalledProcessError(result.returncode, full_cmd, output=result.stdout, stderr=result.stderr)
            # Return the CompletedProcess object for popen=False
            return result

    except subprocess.CalledProcessError as e:
        print(f"!!! Command failed with exit code {e.returncode} !!! Logs are in the files above.", flush=True)
        if check: raise # Re-raise if check=True
        # If check=False, return the failed result object
        if not popen: return e.cmd # A bit hacky, maybe return None or the exception itself? Let's return the result obj.
        return None # Should not happen for popen=True path unless Popen fails?
    except FileNotFoundError as e:
        print(f"!!! Command or Conda environment not found: {e}. Is conda installed/in PATH? Is env '{env_name}' correct? Check 'conda_base_env' or ensure conda command works. !!!", flush=True)
        if popen and not check: return None
        if not popen:
             # Create a dummy result for FileNotFoundError if popen=False
             return subprocess.CompletedProcess(args=full_cmd, returncode=-1, stdout=f"FileNotFoundError: {e}", stderr="")
        raise # Re-raise if popen=True or check=True
    except Exception as e:
        print(f"!!! An unexpected error occurred while running command: {e} !!!", flush=True)
        if popen and not check: return None
        if not popen:
            # Create a dummy result for other exceptions if popen=False
            return subprocess.CompletedProcess(args=full_cmd, returncode=-2, stdout=f"Exception: {e}", stderr=traceback.format_exc())
        raise # Re-raise if popen=True or check=True
    finally:
        # Close files only if command ran synchronously
        if not popen:
            if stdout_f: stdout_f.close()
            if stderr_f: stderr_f.close()

def kill_process_group_and_close_logs(
    process: Optional[subprocess.Popen],
    stdout_log_f: Optional[IO[Any]],
    stderr_log_f: Optional[IO[Any]]
):
    """Reliably kills the process group and closes associated log file handles."""
    if process and process.poll() is None:
        pgid = 0
        try:
            pgid = os.getpgid(process.pid)
            print(f"Attempting to kill process group {pgid} (PID: {process.pid}) (SIGTERM)...", flush=True)
            # Ensure log files are flushed before killing
            if stdout_log_f and not stdout_log_f.closed: stdout_log_f.flush()
            if stderr_log_f and not stderr_log_f.closed: stderr_log_f.flush()
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(5) # Give it time to shut down gracefully

            if process.poll() is None: # Check if it terminated
                print(f"Process group {pgid} did not exit via SIGTERM, sending SIGKILL...", flush=True)
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(2) # Give it time to react to SIGKILL

            # Wait for the process to avoid zombies, with a timeout
            try:
                process.wait(timeout=10)
                print(f"Process group {pgid} terminated (exit code: {process.returncode}).", flush=True)
            except subprocess.TimeoutExpired:
                 print(f"Warning: Timeout waiting for process group {pgid} to terminate after SIGKILL.", flush=True)

        except ProcessLookupError:
            print(f"Process group {pgid} (PID: {process.pid}) already gone.", flush=True)
        except Exception as e:
            print(f"Error killing process group {pgid} (PID: {process.pid}): {e}", flush=True)
            # Fallback kill attempt on the main process PID if group kill failed
            try:
                if process.poll() is None: process.terminate()
                time.sleep(2)
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
                print(f"Main process PID {process.pid} terminated (fallback).")
            except Exception as fallback_e:
                 print(f"Error during fallback kill of PID {process.pid}: {fallback_e}", flush=True)

    elif process:
         # Handle case where process terminated before kill attempt but object exists
         exit_code_str = f"(exit code: {process.returncode})" if hasattr(process, 'returncode') and process.returncode is not None else ""
         print(f"Server process (PID: {process.pid}) already terminated {exit_code_str} before kill attempt.", flush=True)
    else:
        print("No server process to kill.", flush=True)

    # Close log files safely (ensure they exist and aren't already closed)
    for f_handle in [stdout_log_f, stderr_log_f]:
        if f_handle and not f_handle.closed:
            try:
                f_handle.close()
            except Exception as e:
                print(f"Error closing log file handle: {e}", flush=True)


# --- New Helper: Check Server Logs for Errors ---
def check_server_logs_for_errors(
    stdout_f: Optional[IO[Any]], # File handle opened in 'r' mode
    stderr_f: Optional[IO[Any]], # File handle opened in 'r' mode
    error_patterns: List[str],
    last_stdout_pos: int,
    last_stderr_pos: int,
) -> Tuple[bool, Optional[str], int, int]:
    """
    Reads new lines from server log files and checks for error patterns using regex.

    Args:
        stdout_f: Open file handle for stdout log.
        stderr_f: Open file handle for stderr log.
        error_patterns: List of regex patterns to search for.
        last_stdout_pos: The file position where the last read ended for stdout.
        last_stderr_pos: The file position where the last read ended for stderr.

    Returns:
        Tuple containing:
        - bool: True if an error pattern was found, False otherwise.
        - Optional[str]: The pattern that matched, or None.
        - int: The new file position for stdout.
        - int: The new file position for stderr.
    """
    new_stdout_pos = last_stdout_pos
    new_stderr_pos = last_stderr_pos
    found_pattern = None

    files_to_check = []
    if stdout_f:
        files_to_check.append((stdout_f, last_stdout_pos, "stdout"))
    if stderr_f:
         files_to_check.append((stderr_f, last_stderr_pos, "stderr"))

    try:
        for f_handle, last_pos, log_type in files_to_check:
            f_handle.seek(last_pos)
            new_lines = f_handle.readlines() # Read only new lines
            current_pos = f_handle.tell() # Update position *after* reading

            if log_type == "stdout":
                new_stdout_pos = current_pos
            else:
                new_stderr_pos = current_pos

            if not new_lines:
                continue

            # print(f"DEBUG: Read {len(new_lines)} new lines from {log_type}", flush=True) # Debug logging
            for line in new_lines:
                if not line.strip(): # Skip empty lines
                    continue
                for pattern in error_patterns:
                    try:
                        # Use re.search for substring matching (regex enabled)
                        if re.search(pattern, line):
                            print(f"  ERROR DETECTED in {log_type}: Pattern '{pattern}' matched line: {line.strip()}", flush=True)
                            found_pattern = pattern
                            # Return immediately on first match
                            return True, found_pattern, new_stdout_pos, new_stderr_pos
                    except re.error as re_err:
                        print(f"  Warning: Invalid regex pattern '{pattern}': {re_err}", flush=True)
                        # Optionally disable this pattern for future checks? For now, just skip.
                        error_patterns.remove(pattern) # Avoid repeated warnings

    except IOError as e:
        print(f"  Warning: IOError reading log file: {e}", flush=True)
        # Return current positions, assuming no error found in this attempt
        return False, None, new_stdout_pos, new_stderr_pos
    except Exception as e:
        print(f"  Warning: Unexpected error checking logs: {e}", flush=True)
        return False, None, new_stdout_pos, new_stderr_pos

    return False, None, new_stdout_pos, new_stderr_pos


# --- Other Helper Functions (_get_model_family, generate_config_hash, get_server_config_for_hashing, get_benchmark_config_for_hashing, check_completion_marker, write_completion_marker) ---
# ... (Include these functions from Version 3, unchanged) ...
def _get_model_family(model_id: str, template: Optional[str]) -> str:
    """Derives a model family name based on the template."""
    if not template: # Default to base name if no template
        template = "{model_id_basename}"

    # Simplistic extraction of the last part of the ID
    model_id_basename = model_id.split('/')[-1]

    try:
        return template.format(
            model_id=model_id,
            model_id_basename=model_id_basename
            # Add more placeholders here if needed
        )
    except KeyError as e:
        print(f"Warning: Invalid placeholder {e} in model.family_template. Using basename '{model_id_basename}'.", flush=True)
        return model_id_basename
    except Exception as e:
        print(f"Warning: Error formatting model.family_template: {e}. Using basename '{model_id_basename}'.", flush=True)
        return model_id_basename

def generate_config_hash(data: Any) -> str:
    """Generates an MD5 hash for a given Python object (dicts, lists, etc.)."""
    # Convert to JSON string with sorted keys for deterministic output
    try:
        serialized_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(serialized_data.encode('utf-8')).hexdigest()
    except TypeError as e:
        print(f"Warning: Could not serialize data for hashing: {e}. Returning fallback hash.", flush=True)
        # Fallback: hash the string representation, less reliable
        return hashlib.md5(str(data).encode('utf-8')).hexdigest()


def get_server_config_for_hashing(engine_name: str, tp: int, pp: int, config: Dict[str, Any]) -> Dict:
    """Extracts the relevant server configuration parts for hashing."""
    engine_def = config.get('engines', {}).get(engine_name, {})
    server_def = engine_def.get('server', {})
    # Include error_patterns in the hash
    return {
        "engine_name": engine_name,
        "tp": tp,
        "pp": pp,
        "model_config": config.get('model', {}),
        "server_global_defaults": config.get('server', {}).get('defaults', {}),
        "engine_server_config": {
            "command_base": server_def.get('command_base', []),
            "args_mapping": server_def.get('args_mapping', {}),
            "defaults": server_def.get('defaults', {}),
            "server_model_arg_is_positional": server_def.get('server_model_arg_is_positional', False),
            "error_patterns": sorted(server_def.get('error_patterns', [])), # Add patterns, sort for consistency
        },
        # Include relevant engine constraints if they affect behavior?
        "engine_constraints": engine_def.get('constraints', {})
        # Add other critical server params if necessary
    }

def get_benchmark_config_for_hashing(
    benchmark_run_config: Dict[str, Any], # The specific entry from benchmark.run
    engine_identifier: str, # e.g., "sglang_tp1_pp1"
    host: str, # Included as it affects connection
    port: int, # Included as it affects connection
    config: Dict[str, Any], # Full config
    resolved_bench_config_path: Optional[str] # Include the potentially resolved path
) -> Dict:
    """Extracts the relevant benchmark configuration parts for hashing."""
    benchmark_type = benchmark_run_config['type']
    bench_def = config.get('benchmark_definitions', {}).get(benchmark_type, {})

    return {
        "benchmark_type": benchmark_type,
        "engine_identifier": engine_identifier, # Links to server config implicitly
        "target_host": host,
        "target_port": port,
        "model_config": config.get('model', {}), # Model affects benchmark behavior
        "benchmark_global_defaults": config.get('benchmark', {}).get('defaults', {}),
        "benchmark_definition": bench_def, # Includes command_base, args_mapping, defaults
        "benchmark_run_overrides": benchmark_run_config.get('overrides', {}),
        "resolved_benchmark_config_path": resolved_bench_config_path, # If applicable
        # Add other critical benchmark params if necessary
    }

def check_completion_marker(marker_path: str, expected_server_hash: str, expected_benchmark_hash: str) -> bool:
    """Checks if a valid completion marker exists with matching hashes."""
    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, 'r') as f:
            marker_data = json.load(f)
        # Check if required keys and hashes match
        if (marker_data.get("server_config_hash") == expected_server_hash and
            marker_data.get("benchmark_config_hash") == expected_benchmark_hash):
            print(f"  Found valid completion marker: {marker_path}", flush=True)
            return True
        else:
            print(f"  Found stale completion marker (hash mismatch): {marker_path}", flush=True)
            return False # Hashes don't match, needs re-run
    except (json.JSONDecodeError, IOError, KeyError, TypeError) as e: # Added TypeError
        print(f"  Found corrupted/invalid completion marker: {marker_path}. Error: {e}", flush=True)
        return False # Problem reading marker, needs re-run

def write_completion_marker(marker_path: str, server_hash: str, benchmark_hash: str):
    """Writes the completion marker file with current hashes and timestamp."""
    marker_data = {
        "server_config_hash": server_hash,
        "benchmark_config_hash": benchmark_hash,
        "completion_timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        print(f"  Wrote completion marker: {marker_path}", flush=True)
    except IOError as e:
        print(f"!!! Error writing completion marker {marker_path}: {e} !!!", flush=True)


# --- Command Generation Functions (get_server_command, get_benchmark_command) ---
# ... (Include these functions from Version 3, unchanged) ...
def get_server_command(
    engine_name: str,
    tp: int,
    pp: int,
    port: int,
    # model_run_dir: str, # No longer strictly needed here
    config: Dict[str, Any]
) -> List[str]:
    """Builds the server start command based on the engine definition in config."""
    # (Identical to previous version - no changes needed for caching logic here)
    if engine_name not in config.get('engines', {}):
        raise ValueError(f"Engine '{engine_name}' not defined in the config file's 'engines' section.")

    engine_def = config['engines'][engine_name]
    server_def = engine_def.get('server', {})
    args_mapping = server_def.get('args_mapping', {})

    # 1. Gather parameter values
    param_values = {
        'tp': tp,
        'pp': pp,
        'port': port,
        'model_id': config['model']['id'],
        'host': config['server']['host'],
    }
    # Merge global server defaults then engine-specific server defaults
    merged_defaults = {**config['server'].get('defaults', {}), **server_def.get('defaults', {})}
    param_values = {**merged_defaults, **param_values} # Specific params override defaults

    # --- Special Handling ---
    if engine_name == "vajra":
        if param_values.get("scheduler") == "FIXED_CHUNK" and 'chunk_size' in args_mapping and 'chunk_size' in param_values:
            args_mapping['chunk_size'] = "--fixed_chunk_replica_scheduler_config_chunk_size {value}"
        if 'prioritizer' in param_values: # Ensure upper case for Vajra
             param_values['prioritizer'] = str(param_values['prioritizer']).upper()
    # --- End Special Handling ---

    # 3. Build the command list
    cmd = list(server_def.get('command_base', []))
    if not cmd:
        raise ValueError(f"Engine '{engine_name}' definition is missing 'server.command_base'.")

    positional_model_arg = None

    # 4. Append arguments based on mapping
    processed_keys = set() # Track keys used in mapping
    # Sort items for deterministic command generation (less critical but good practice)
    for key, template in sorted(args_mapping.items()):
        if key in param_values and param_values[key] is not None:
            value = param_values[key]
            processed_keys.add(key)

            if "{value...}" in template: # Handle list expansion
                if isinstance(value, list):
                    base_arg = template.replace("{value...}", "").strip()
                    if base_arg: cmd.append(base_arg)
                    cmd.extend(map(str, value))
                else:
                    print(f"Warning: Arg mapping for '{key}' expects a list ('{{value...}}') but got type {type(value)}. Ignoring.", flush=True)
            elif "{value}" in template: # Handle single value substitution
                try:
                    formatted_arg = shlex.split(template.format(value=str(value)))
                    cmd.extend(formatted_arg)
                except Exception as format_e:
                     print(f"Warning: Could not format arg for key '{key}' with template '{template}' and value '{value}': {format_e}", flush=True)
            elif isinstance(value, bool) and value: # Handle boolean flag (True case)
                cmd.append(template)
            elif isinstance(value, bool) and not value and template.startswith("--no-"): # Handle boolean flag (False case with --no- prefix)
                 cmd.append(template)
            elif not isinstance(value, bool): # Non-boolean without {value} template
                 # This condition might be too strict depending on the flag conventions
                 # print(f"Warning: Argument template '{template}' for key '{key}' has no '{{value}}' placeholder but value ('{value}') is not boolean True. Ignoring.", flush=True)
                 pass # Allow flags without value if non-boolean? Revisit if needed.

    # 5. Handle positional model argument
    if server_def.get('server_model_arg_is_positional', False):
         if 'model_id' in param_values:
             positional_model_arg = str(param_values['model_id'])
             processed_keys.add('model_id') # Mark as used
         else:
             print(f"Warning: Engine '{engine_name}' expects positional model_id, but 'model_id' not found.", flush=True)

    # Append positional model argument at the end if specified
    if positional_model_arg:
         cmd.append(positional_model_arg)

    return cmd

def get_benchmark_command(
    benchmark_config: Dict[str, Any], # The specific entry from benchmark.run list
    engine_identifier: str, # e.g., "sglang_tp1_pp1"
    host: str,
    port: int,
    run_base_dir: str, # Base directory for the engine/tp/pp run
    config: Dict[str, Any], # Full config
    main_config_path: str # Path to the loaded config file
) -> Tuple[List[str], str, str, Optional[str], Optional[str]]: # Returns command, log_dir, log_prefix, conda_env_name, resolved_bench_config_path
    """Builds a benchmark command based on its definition in config."""

    benchmark_type = benchmark_config['type']
    if benchmark_type not in config.get('benchmark_definitions', {}):
        raise ValueError(f"Benchmark type '{benchmark_type}' not defined in benchmark_definitions.")

    bench_def = config['benchmark_definitions'][benchmark_type]
    args_mapping = bench_def.get('args_mapping', {})

    # 1. Determine output directory and log paths for *this* benchmark
    # Place benchmark logs inside the engine-specific run directory
    bench_log_dir = os.path.join(run_base_dir, f"benchmark_{benchmark_type}")
    log_prefix = f"{benchmark_type}_benchmark" # Filename prefix

    # 2. Gather parameter values
    # Start with runner-provided dynamic values
    param_values = {
        'host': host,
        'port': port,
        'model_id': config['model']['id'],
        'engine_identifier': engine_identifier,
        'output_dir': bench_log_dir, # Pass the specific dir for this benchmark instance
        'api_base': f"http://{host}:{port}/v1", # Common default, might be overridden if needed
    }

    # Merge defaults: Global benchmark defaults -> Benchmark definition defaults -> Run-specific overrides
    merged_defaults = {
        **config['benchmark'].get('defaults', {}),
        **bench_def.get('defaults', {}),
        **benchmark_config.get('overrides', {})
    }
    param_values = {**merged_defaults, **param_values} # Runner-provided values override defaults/overrides

    # --- Special Handling: Benchmark Internal Config Path ---
    resolved_bench_config_path: Optional[str] = None # Initialize
    if 'config_path_template' in param_values and 'config_path' in args_mapping:
        template = param_values['config_path_template']
        model_family = _get_model_family(config['model']['id'], config['model'].get('family_template'))
        config_dir = os.path.dirname(main_config_path) if main_config_path else "."

        try:
            resolved_path_str = template.format(
                model_family=model_family,
                model_id=config['model']['id'],
                config_dir=config_dir
                # Add more placeholders if needed
            )
            # Resolve relative paths based on the main config file's directory
            resolved_path = Path(config_dir) / resolved_path_str
            resolved_bench_config_path = str(resolved_path.resolve()) # Use absolute path
            param_values['config_path'] = resolved_bench_config_path # Update param for command gen
            print(f"  Resolved benchmark config path: {param_values['config_path']}", flush=True)
        except KeyError as e:
            print(f"Warning: Invalid placeholder {e} in benchmark config_path_template '{template}'. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None) # Remove if resolution failed
        except Exception as e:
            print(f"Warning: Error resolving benchmark config_path_template '{template}': {e}. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None)
    elif 'config_path' in param_values: # Handle case where config_path is directly provided (e.g., in overrides)
        # Ensure it's absolute for consistency in hashing/logging
        config_dir = os.path.dirname(main_config_path) if main_config_path else "."
        potential_path = Path(config_dir) / param_values['config_path']
        if potential_path.is_absolute():
             resolved_bench_config_path = param_values['config_path']
             # param_values['config_path'] already absolute
        elif potential_path.exists():
            resolved_bench_config_path = str(potential_path.resolve())
            param_values['config_path'] = resolved_bench_config_path # Update param for command gen
        else:
            resolved_bench_config_path = param_values['config_path'] # Keep as is if not found relative to config dir
            print(f"  Note: Benchmark config_path '{resolved_bench_config_path}' not found relative to main config, using as is.", flush=True)


    # --- End Special Handling ---


    # 3. Build the command list
    cmd = list(bench_def.get('command_base', []))
    if not cmd:
        raise ValueError(f"Benchmark definition '{benchmark_type}' is missing 'command_base'.")

    # 4. Append arguments based on mapping
    processed_keys = set()
    # Sort items for deterministic command generation
    for key, template in sorted(args_mapping.items()):
        if key in param_values and param_values[key] is not None:
            value = param_values[key]
            processed_keys.add(key) # Mark key as used by mapping

            # --- Handle different template types ---
            if template == "{value}": # Special case: just the value
                cmd.append(str(value))
            elif "{value...}" in template: # List expansion
                 if isinstance(value, list):
                    base_arg = template.split('{value...}')[0].strip()
                    if base_arg:
                        cmd.append(base_arg)
                    cmd.extend(map(str, value))
                 else:
                    print(f"Warning: Arg mapping for '{key}' expects list ('{{value...}}') but got {type(value)}. Ignoring.", flush=True)

            elif "{value}" in template: # Standard substitution
                try:
                    formatted_arg = shlex.split(template.format(value=str(value)))
                    cmd.extend(formatted_arg)
                except Exception as format_e:
                     print(f"Warning: Could not format benchmark arg for key '{key}' with template '{template}' and value '{value}': {format_e}", flush=True)

            elif isinstance(value, bool): # Boolean flags
                 if value is True:
                     cmd.append(template)
                 elif value is False and template.startswith("--no-"):
                      cmd.append(template)
                 # else: False, non --no- flag => omit

            # Handle simple presence flags (e.g., a flag defined in mapping that doesn't depend on a value)
            # This logic needs refinement based on how such flags are represented.
            # Example: If args_mapping has "enable_feature": "--enable-it" and param_values['enable_feature'] is True
            elif not isinstance(value, bool) and not ("{value}" in template or "{value...}" in template):
                 # This assumes the template *is* the flag and the key's presence implies adding it.
                 # Might need a more explicit way to define these flags.
                 # Let's assume for now simple key presence maps to the flag template directly if no {value} placeholder
                 print(f"  Note: Adding flag '{template}' based on presence of key '{key}' (value: {value})", flush=True)
                 cmd.append(template)


    # Determine conda environment for the benchmark
    bench_env_pattern = bench_def.get('environment') # Can be null/None
    bench_conda_env_name = None
    if bench_env_pattern:
         try:
             # Use runner's env_prefix if available
             env_prefix = config.get('runner', {}).get('env_prefix', '')
             bench_conda_env_name = bench_env_pattern.format(env_prefix=env_prefix, benchmark_type=benchmark_type)
         except KeyError as e:
             print(f"Warning: Invalid placeholder {e} in benchmark environment pattern for {benchmark_type}.", flush=True)
         except Exception as e:
              print(f"Warning: Error formatting benchmark environment pattern '{bench_env_pattern}': {e}.", flush=True)


    # Optional: Warn about unused parameters (commented out)
    # internal_params = {'config_path_template'}
    # unused_params = set(param_values.keys()) - processed_keys - internal_params
    # if unused_params:
    #     print(f"Note: The following benchmark parameters were available but not used by mappings for {benchmark_type}: {unused_params}", flush=True)

    return cmd, bench_log_dir, log_prefix, bench_conda_env_name, resolved_bench_config_path


# --- Config Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file and performs validation."""
    print(f"Loading configuration from: {config_path}", flush=True)
    abs_config_path = str(Path(config_path).resolve()) # Get absolute path
    try:
        with open(abs_config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError("Config file must contain a YAML dictionary.")
        print("Configuration loaded successfully.", flush=True)
        # Basic validation (can be expanded)
        required_keys = ['runner', 'model', 'server', 'benchmark', 'engines', 'run_matrix', 'benchmark_definitions']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required top-level key in config: '{key}'")
        # Further validation
        if not isinstance(config.get('benchmark', {}).get('run'), list):
             raise ValueError("'benchmark.run' must be a list in the config.")
        if not isinstance(config.get('benchmark_definitions'), dict):
             raise ValueError("'benchmark_definitions' must be a dictionary.")

        # Validate benchmark types listed in benchmark.run are defined
        defined_benchmarks = set(config['benchmark_definitions'].keys())
        for run_conf in config['benchmark']['run']:
            if not isinstance(run_conf, dict) or 'type' not in run_conf:
                 raise ValueError("Each item in 'benchmark.run' must be a dictionary with a 'type' key.")
            if run_conf['type'] not in defined_benchmarks:
                 raise ValueError(f"Benchmark type '{run_conf['type']}' listed in 'benchmark.run' is not defined in 'benchmark_definitions'.")

        # Validate engines in run_matrix are defined
        defined_engines = set(config.get('engines', {}).keys())
        for engine_to_run in config['run_matrix'].get('engines', []):
            if engine_to_run not in defined_engines:
                 raise ValueError(f"Engine '{engine_to_run}' listed in 'run_matrix' is not defined in the 'engines' section.")

        # Validate error_patterns if present
        for engine_name, engine_data in config.get('engines', {}).items():
            error_patterns = engine_data.get('server', {}).get('error_patterns')
            if error_patterns is not None:
                if not isinstance(error_patterns, list):
                    raise ValueError(f"Engine '{engine_name}' server.error_patterns must be a list of strings.")
                for item in error_patterns:
                    if not isinstance(item, str):
                        raise ValueError(f"Engine '{engine_name}' server.error_patterns must contain only strings.")

        # Store absolute path for later use (e.g., resolving relative paths)
        config['_main_config_path'] = abs_config_path
        return config
    except FileNotFoundError:
        print(f"!!! Error: Config file not found at {config_path} (resolved to {abs_config_path}) !!!", flush=True)
        raise
    except yaml.YAMLError as e:
        print(f"!!! Error parsing YAML config file {abs_config_path}: {e} !!!", flush=True)
        raise
    except ValueError as e:
        print(f"!!! Config file validation error: {e} !!!", flush=True)
        raise
    except Exception as e:
        print(f"!!! An unexpected error occurred loading config: {e} !!!", flush=True)
        traceback.print_exc()
        raise


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run LLM Engine Benchmarks using Pluggable YAML Config Definitions")
    parser.add_argument("--config", type=str, default="./scripts/benchmark_config.yml", help="Path to the YAML configuration file.") # Default to config.yaml
    parser.add_argument("--force-rerun", action="store_true", help="Ignore completion markers and force all benchmarks to run.")
    args = parser.parse_args()

    try: # Wrap main logic in try/except for config loading errors
        config = load_config(args.config)
    except Exception:
        # Error already printed by load_config
        exit(1) # Exit if config fails to load/validate

    main_config_path = config['_main_config_path'] # Get resolved path

    # Extract config sections
    runner_config = config['runner']
    server_config = config['server']
    benchmark_control = config['benchmark'] # Contains the 'run' list
    run_matrix = config['run_matrix']
    engine_definitions = config['engines']

    base_output_dir = runner_config['base_output_dir']
    os.makedirs(base_output_dir, exist_ok=True)

    current_port = server_config['start_port']
    parallel_combinations = list(itertools.product(run_matrix.get('tp_dims', [1]), run_matrix.get('pp_dims', [1])))
    env_prefix = runner_config.get('env_prefix', '')

    # --- Main Loop ---
    for engine_name in run_matrix.get('engines', []):
        engine_def = engine_definitions.get(engine_name)
        if not engine_def: continue # Should have been caught by validation, but safe check
        engine_constraints = engine_def.get('constraints', {})

        for tp, pp in parallel_combinations:
            # Check GPU/Constraint requirements
            gpus_required = tp * pp
            max_gpus = runner_config.get('max_gpus', 999)
            if gpus_required > max_gpus:
                print(f"\n--- Skipping {engine_name} TP={tp}, PP={pp} (requires {gpus_required} GPUs > max {max_gpus}) ---\n", flush=True)
                continue
            max_pp_allowed = engine_constraints.get('max_pp')
            if max_pp_allowed is not None and pp > max_pp_allowed:
                 print(f"\n--- Skipping {engine_name} TP={tp}, PP={pp} (PP > max {max_pp_allowed} for this engine) ---\n", flush=True)
                 continue

            engine_tp_pp_id = f"{engine_name}_tp{tp}_pp{pp}"
            run_base_dir = os.path.join(base_output_dir, engine_tp_pp_id) # Specific dir for this engine/tp/pp combo

            # Determine server conda environment name
            server_env_pattern = engine_def.get('environment', {}).get('conda_env_name')
            server_conda_env_name = None
            if server_env_pattern:
                try:
                    server_conda_env_name = server_env_pattern.format(env_prefix=env_prefix, engine_name=engine_name)
                except Exception as e:
                     print(f"Warning: Failed to format server conda env name for {engine_name}: {e}", flush=True)

            # --- Calculate Server Config Hash (once per engine/tp/pp) ---
            server_config_data = get_server_config_for_hashing(engine_name, tp, pp, config)
            server_config_hash = generate_config_hash(server_config_data)
            print(f"--- Server Config Hash for {engine_tp_pp_id}: {server_config_hash} ---", flush=True)
            # --- End Server Hash Calculation ---

            print(f"\n{'='*25} Starting Run: {engine_tp_pp_id} {'='*25}\n", flush=True)
            server_process = None
            # Ensure file handles are always initialized to None
            server_stdout_f, server_stderr_f = None, None
            server_launch_failed = False # Flag specific to Popen failure
            detected_fatal_error = False # Flag for detected error patterns
            server_ready_status = False # Flag for API readiness

            try:
                # 1. Setup Server Logging Dir
                server_log_dir = os.path.join(run_base_dir, "server_logs")
                server_stdout_log, server_stderr_log = setup_logging(server_log_dir, "server")

                # 2. Clean up temp dirs
                if runner_config.get('cleanup_tmp', True):
                    print(f"--- Cleaning up temporary LLM workspace directories ---", flush=True)
                    run_command(["rm", "-rf", "/tmp/*-llm-workspace"], check=False, popen=False) # run_command handles errors

                # 3. Construct & Start Server Command
                print(f"--- Launching {engine_name} Server (TP={tp}, PP={pp}) ---", flush=True)
                server_cmd_list = get_server_command(
                    engine_name=engine_name, tp=tp, pp=pp,
                    port=current_port, config=config
                )
                # Reset file handles before Popen attempt
                server_stdout_f, server_stderr_f = None, None
                server_run_result = run_command(
                    server_cmd_list,
                    env_name=server_conda_env_name,
                    conda_base_env=runner_config.get('conda_base_env'),
                    popen=True, check=False, # check=False to handle startup failures gracefully
                    stdout_log_path=server_stdout_log, stderr_log_path=server_stderr_log
                )

                if server_run_result is None:
                    # run_command already printed error, just set flag and proceed to finally
                    server_launch_failed = True
                    raise RuntimeError(f"Failed to start server process for {engine_tp_pp_id}. Check logs or conda setup ({server_conda_env_name}).")

                server_process, server_stdout_f, server_stderr_f = server_run_result

                # Small delay to allow process start and initial log output
                time.sleep(3)

                # --- Combined Readiness & Error Check Loop ---
                print(f"--- Waiting for server readiness & monitoring logs ({server_config['readiness_timeout']}s timeout) ---", flush=True)
                start_time = time.monotonic()
                error_patterns = engine_def.get('server', {}).get('error_patterns', [])
                last_stdout_pos = 0
                last_stderr_pos = 0
                read_stdout_f, read_stderr_f = None, None

                try:
                    # Open logs for reading *after* Popen has opened them for writing
                    # Add checks for file existence before opening
                    if os.path.exists(server_stdout_log):
                        read_stdout_f = open(server_stdout_log, 'r')
                    else:
                         print(f"  Warning: Server stdout log not found immediately: {server_stdout_log}", flush=True)
                    if os.path.exists(server_stderr_log):
                        read_stderr_f = open(server_stderr_log, 'r')
                    else:
                         print(f"  Warning: Server stderr log not found immediately: {server_stderr_log}", flush=True)

                    while time.monotonic() - start_time < server_config['readiness_timeout']:
                        # Check 1: Process died unexpectedly?
                        if server_process.poll() is not None:
                            raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) exited prematurely (code {server_process.returncode}) during readiness/error check. Logs: {server_log_dir}")

                        # Check 2: Errors in logs? (Only if patterns are defined)
                        if error_patterns and (read_stdout_f or read_stderr_f):
                             found_error, error_msg, last_stdout_pos, last_stderr_pos = check_server_logs_for_errors(
                                 read_stdout_f, read_stderr_f, error_patterns, last_stdout_pos, last_stderr_pos
                             )
                             if found_error:
                                 detected_fatal_error = True
                                 print(f"\n!!! Detected fatal server error pattern: '{error_msg}' in logs for {engine_tp_pp_id} !!!", flush=True)
                                 break # Exit the readiness loop immediately

                        # Check 3: API Ready? (Use a simple request check)
                        api_check_url = f"http://{server_config['host']}:{current_port}/v1/models" # Adjust endpoint if needed
                        try:
                            response = requests.get(api_check_url, timeout=2) # Short timeout for check
                            if response.status_code == 200:
                                print(f"\nServer API is ready! ({api_check_url} responded {response.status_code} in {time.monotonic() - start_time:.2f}s)", flush=True)
                                server_ready_status = True
                                break # Exit the readiness loop
                            else:
                                print(f"S({response.status_code})", end='', flush=True)
                        except ConnectionError:
                            print(".", end='', flush=True)
                        except Timeout:
                            print("T", end='', flush=True)
                        except Exception as api_e:
                            print(f"E({type(api_e).__name__})", end='', flush=True)

                        # Wait before next check cycle
                        time.sleep(2)

                finally:
                    # Ensure read handles are closed
                    if read_stdout_f: read_stdout_f.close()
                    if read_stderr_f: read_stderr_f.close()
                # --- End of Readiness & Error Check Loop ---

                # --- Post-Loop Handling ---
                if detected_fatal_error:
                    print(f"--- Skipping benchmarks for {engine_tp_pp_id} due to detected server error during startup. ---", flush=True)
                    # Let finally block handle cleanup
                elif not server_ready_status:
                     # Timeout occurred without readiness or fatal error
                     if server_process.poll() is None: # Check if process is still running
                         raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) did not become ready within {server_config['readiness_timeout']}s and no fatal errors detected. Server might be hung. Logs: {server_log_dir}")
                     else: # Process died during timeout period but after initial check
                          raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) exited (code {server_process.returncode}) during timeout period without becoming ready. Logs: {server_log_dir}")
                else:
                    # Server is ready and no fatal errors detected during startup check
                    print("--- Server is running and ready, proceeding to benchmarks ---", flush=True)

                    # --- Benchmark Execution Loop ---
                    stream_bench_logs = runner_config.get('stream_logs', False)
                    for bench_run_config in benchmark_control.get('run', []):
                        if not bench_run_config.get('enabled', False):
                            print(f"\n--- Skipping Benchmark: {bench_run_config['type']} (disabled in config) ---", flush=True)
                            continue

                        benchmark_type = bench_run_config['type']
                        print(f"\n--- Preparing Benchmark: {benchmark_type} for {engine_tp_pp_id} ---", flush=True)

                        # benchmark_succeeded = False # Reset for each benchmark
                        try:
                            # Get benchmark command, paths, env, resolved config path
                            bench_cmd, bench_log_dir, bench_log_prefix, bench_conda_env, resolved_bench_config_path = get_benchmark_command(
                                benchmark_config=bench_run_config,
                                engine_identifier=engine_tp_pp_id,
                                host=server_config['host'], port=current_port,
                                run_base_dir=run_base_dir,
                                config=config,
                                main_config_path=main_config_path
                            )

                            # Calculate Benchmark Config Hash
                            bench_config_data = get_benchmark_config_for_hashing(
                                bench_run_config, engine_tp_pp_id, server_config['host'], current_port, config, resolved_bench_config_path
                            )
                            benchmark_config_hash = generate_config_hash(bench_config_data)
                            print(f"  Benchmark Config Hash: {benchmark_config_hash}", flush=True)

                            # Define completion marker path
                            completion_marker_path = os.path.join(bench_log_dir, COMPLETION_MARKER_FILENAME)

                            # Check for Completion Marker
                            if not args.force_rerun:
                                if check_completion_marker(completion_marker_path, server_config_hash, benchmark_config_hash):
                                    print(f"--- Skipping Benchmark: {benchmark_type} for {engine_tp_pp_id} (already completed with matching config) ---", flush=True)
                                    continue # Skip to the next benchmark type
                                else:
                                    pass # Proceed with run
                            else:
                                print("  --force-rerun specified, ignoring completion marker check.", flush=True)

                            # Setup benchmark logging
                            bench_stdout_log, bench_stderr_log = setup_logging(bench_log_dir, bench_log_prefix)
                            final_conda_env = bench_conda_env if bench_conda_env is not None else server_conda_env_name

                            # Execute benchmark command
                            print(f"--- Running Benchmark: {benchmark_type} for {engine_tp_pp_id} ---", flush=True)
                            benchmark_result = run_command(
                                bench_cmd,
                                env_name=final_conda_env,
                                conda_base_env=runner_config.get('conda_base_env'),
                                popen=False,
                                check=False, # Check return code below
                                stdout_log_path=bench_stdout_log,
                                stderr_log_path=bench_stderr_log,
                                stream_logs=stream_bench_logs
                            )

                            # Write Completion Marker on Success
                            if benchmark_result and benchmark_result.returncode == 0:
                                print(f"--- Benchmark {benchmark_type} completed successfully ---", flush=True)
                                write_completion_marker(completion_marker_path, server_config_hash, benchmark_config_hash)
                                # benchmark_succeeded = True
                            else:
                                rc = benchmark_result.returncode if hasattr(benchmark_result, 'returncode') else "N/A"
                                print(f"!!! Benchmark '{benchmark_type}' failed or did not run correctly for {engine_tp_pp_id} (exit code: {rc}) !!!", flush=True)
                                # Do NOT write marker on failure

                        except Exception as bench_e:
                            print(f"!!! An error occurred during benchmark '{benchmark_type}' setup or execution for {engine_tp_pp_id}: {bench_e} !!!", flush=True)
                            traceback.print_exc()
                            # Continue to next benchmark type

            except Exception as e:
                # Catch errors during server launch, readiness check, or benchmark loop setup
                print(f"\n!!! === ERROR during run for {engine_tp_pp_id}: {e} === !!!\n", flush=True)
                # Avoid double printing stack trace if run_command already did
                if not isinstance(e, (subprocess.CalledProcessError, FileNotFoundError)):
                     traceback.print_exc()


            finally:
                # --- Cleanup ---
                print(f"\n--- Cleaning up server for {engine_tp_pp_id} ---", flush=True)
                # Pass the Popen object and the *write* file handles to the cleanup function
                kill_process_group_and_close_logs(server_process, server_stdout_f, server_stderr_f)
                # Reset state variables for the next loop iteration
                server_process, server_stdout_f, server_stderr_f = None, None, None
                server_launch_failed = False
                detected_fatal_error = False
                server_ready_status = False

                print(f"\n{'='*25} Finished Run: {engine_tp_pp_id} {'='*25}\n", flush=True)
                current_port += 1 # Increment port regardless of success/failure for this combo
                time.sleep(5) # Pause between different engine/tp/pp runs

    print("\nAll configured benchmark runs completed, skipped, or failed.", flush=True)


if __name__ == "__main__":
    main()