# -*- coding: utf-8 -*-
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
import sys # For sys.stdout/stderr
import threading # For real-time log streaming
import contextlib # For stdout redirection context manager

# --- Constants ---
COMPLETION_MARKER_FILENAME = ".benchmark_complete"

# --- Helper: Context Manager for Silencing ---
@contextlib.contextmanager
def redirect_stdout_stderr(target_stdout=os.devnull, target_stderr=os.devnull):
    """Temporarily redirects stdout and/or stderr."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        # Open the target files/devices
        stdout_f = open(target_stdout, 'w')
        stderr_f = open(target_stderr, 'w')
        # Redirect
        sys.stdout = stdout_f
        sys.stderr = stderr_f
        yield # Allow code within the 'with' block to run
    except Exception as e:
        # Print exceptions during redirection setup TO THE ORIGINAL stderr
        print(f"Error during stream redirection: {e}", file=original_stderr)
        traceback.print_exc(file=original_stderr)
        # If yield fails, the exception propagates outwards normally
        raise
    finally:
        # Ensure streams are restored and files closed
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if 'stdout_f' in locals() and stdout_f:
            try: stdout_f.close()
            except Exception: pass
        if 'stderr_f' in locals() and stderr_f:
            try: stderr_f.close()
            except Exception: pass

# --- Helper Functions (setup_logging, _log_stream_reader, run_command, kill_process_group, etc.) ---

def setup_logging(log_dir: str, name_prefix: str) -> Tuple[str, str]:
    """Creates log directory and returns paths for stdout and stderr log files."""
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, f"{name_prefix}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{name_prefix}_stderr.log")
    # Ensure fresh logs for this specific run command invocation
    for log_path in [stdout_log_path, stderr_log_path]:
        try:
            with open(log_path, 'w') as f:
                f.write("") # Truncate the file
        except IOError as e:
             print(f"Warning: Could not truncate log file {log_path}: {e}", file=sys.stderr, flush=True) # Use stderr for warnings
    return stdout_log_path, stderr_log_path

# --- New Thread Helper for Streaming ---
def _log_stream_reader(
    stream: Optional[IO[bytes]], # Process pipe (e.g., process.stdout)
    log_path: Optional[str],
    console_stream: Optional[IO[str]], # sys.stdout or sys.stderr
    stream_name: str # "stdout" or "stderr" for logging messages
) -> None:
    """
    Reads from a stream (process pipe), writes to a log file,
    and optionally prints to a console stream in real-time.
    """
    log_f = None
    try:
        if log_path:
            # Open in append bytes mode. setup_logging should have truncated it.
            log_f = open(log_path, 'ab')

        if stream:
            # Read bytes line by line.
            for line_bytes in iter(stream.readline, b''):
                if log_f:
                    try:
                        log_f.write(line_bytes)
                        log_f.flush() # Ensure it's written immediately
                    except Exception as write_e:
                         print(f"\n!!! Error writing to {stream_name} log {log_path}: {write_e} !!!\n", file=sys.stderr, flush=True)

                if console_stream:
                    try:
                        # Decode for console, replacing errors
                        line_str = line_bytes.decode('utf-8', errors='replace')
                        print(line_str, end='', file=console_stream, flush=True)
                    except Exception as print_e:
                         print(f"\n!!! Error printing {stream_name} to console: {print_e} !!!\n", file=sys.stderr, flush=True)

    except Exception as e:
        print(f"\n!!! Error in {stream_name} log reader thread ({log_path}): {e} !!!\n", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
    finally:
        if stream and not stream.closed:
            try: stream.close()
            except Exception: pass
        if log_f and not log_f.closed:
            try: log_f.close()
            except Exception: pass

# --- Modified run_command (Using Popen + Threads) ---
def run_command(
    cmd_list: List[str],
    env_name: Optional[str] = None,
    conda_base_env: Optional[str] = None,
    stdout_log_path: Optional[str] = None,
    stderr_log_path: Optional[str] = None,
    stream_logs: bool = False, # Controls console mirroring
    check_return_code: bool = True, # Controls if caller wants immediate check + raise
) -> Tuple[Optional[subprocess.Popen], Optional[threading.Thread], Optional[threading.Thread]]:
    """
    Runs a command asynchronously using Popen, streams logs in real-time
    to files and optionally to console using threads.
    Returns Tuple (Popen object, stdout reader thread or None, stderr reader thread or None).
    Raises exceptions on failure if check_return_code is True or during Popen setup.
    """
    env_vars = os.environ.copy()
    full_cmd: List[str] = []

    # --- Conda Activation Logic ---
    if env_name:
        conda_base = conda_base_env or os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_ROOT")
        if not conda_base and env_name:
            try:
                conda_info_cmd = ["conda", "info", "--base"]
                result = subprocess.run(conda_info_cmd, capture_output=True, text=True, check=True, timeout=5)
                conda_base = result.stdout.strip()
                print(f"  Auto-detected conda base: {conda_base}", flush=True)
            except Exception:
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
                    conda_base = None
                    print(f"  Error: Cannot auto-detect or guess conda base. Set 'conda_base_env' in config or ensure conda is in PATH.", flush=True)

        conda_env_path = None
        if conda_base:
            conda_env_path = Path(conda_base) / "envs" / env_name
            lib_path = conda_env_path / "lib"
            if lib_path.is_dir():
                existing_ld_path = env_vars.get("LD_LIBRARY_PATH", "")
                env_vars["LD_LIBRARY_PATH"] = f"{lib_path}:{existing_ld_path}" if existing_ld_path else str(lib_path)

        if conda_env_path and conda_env_path.exists():
             full_cmd = ["conda", "run", "--no-capture-output", "--prefix", str(conda_env_path)] + cmd_list
        elif env_name and conda_base:
             print(f"  Warning: Conda env path for '{env_name}' not found, attempting activation by name.", flush=True)
             full_cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd_list
        elif env_name:
             print(f"  Error: Cannot activate conda env '{env_name}' due to missing base path. Command will run in current env.", flush=True)
             full_cmd = cmd_list
        else: # No env name provided, shouldn't happen if conda_base was needed
            full_cmd = cmd_list
    else:
        full_cmd = cmd_list
    # --- End Conda Activation Logic ---

    cmd_str = ' '.join(shlex.quote(str(part)) for part in full_cmd)
    print(f"\nExecuting{' in env ' + shlex.quote(env_name) if env_name else ''}: {cmd_str}", flush=True)
    if stdout_log_path: print(f"  stdout log: {stdout_log_path}", flush=True)
    if stderr_log_path: print(f"  stderr log: {stderr_log_path}", flush=True)

    process: Optional[subprocess.Popen] = None
    stdout_thread: Optional[threading.Thread] = None
    stderr_thread: Optional[threading.Thread] = None

    try:
        # Start the subprocess
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid, # For process group killing
            env=env_vars
        )

        # Start reader threads
        stdout_thread = threading.Thread(
            target=_log_stream_reader,
            args=(process.stdout, stdout_log_path, sys.stdout if stream_logs else None, "stdout"),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=_log_stream_reader,
            args=(process.stderr, stderr_log_path, sys.stderr if stream_logs else None, "stderr"),
            daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        # Optional: Wait and check return code immediately
        if check_return_code:
            return_code = process.wait() # Wait for process to finish
            # Wait for threads to finish writing logs *after* process ends
            if stdout_thread: stdout_thread.join(timeout=5)
            if stderr_thread: stderr_thread.join(timeout=5)
            print(f"Command finished with exit code {return_code}.", flush=True) # Log exit code
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, full_cmd)
            # Return process (already finished) and threads (already joined)
            return process, stdout_thread, stderr_thread
        else:
            # Return immediately, caller is responsible for waiting/joining
            return process, stdout_thread, stderr_thread

    except FileNotFoundError as e:
        print(f"!!! Command or Conda environment not found: {e}. Is conda installed/in PATH? Is env '{env_name}' correct? Check 'conda_base_env' or ensure conda command works. !!!", flush=True)
        # Manually call cleanup for started threads/process if possible? Unlikely path here.
        raise # Re-raise FileNotFoundError
    except subprocess.CalledProcessError as e:
        # This block is now only reached if check_return_code=True
        print(f"!!! Command failed with exit code {e.returncode} !!! Logs should be in the files above.", flush=True)
        raise # Re-raise CalledProcessError
    except Exception as e:
        print(f"!!! An unexpected error occurred while starting command or threads: {e} !!!", flush=True)
        traceback.print_exc()
        # Attempt cleanup if process started but threads failed?
        if process and process.poll() is None:
            try:
                 kill_process_group(process) # Use simplified kill here
                 process.wait(timeout=5)
            except Exception: pass
        # Don't return threads if they didn't start properly
        raise # Re-raise the unexpected error

# --- Updated kill_process_group (no longer needs file handles) ---
def kill_process_group(process: Optional[subprocess.Popen]):
    """Reliably kills the process group associated with the Popen object."""
    if process and process.poll() is None:
        pgid = 0
        try:
            pgid = os.getpgid(process.pid)
            print(f"Attempting to kill process group {pgid} (PID: {process.pid}) (SIGTERM)...", flush=True)
            os.killpg(pgid, signal.SIGTERM)
            # Give it time to shut down gracefully
            process.wait(timeout=10) # Wait with timeout after SIGTERM
            print(f"Process group {pgid} terminated via SIGTERM (exit code: {process.returncode}).", flush=True)

        except subprocess.TimeoutExpired:
            print(f"Process group {pgid} did not exit via SIGTERM within timeout, sending SIGKILL...", flush=True)
            try:
                # Ensure pgid is still valid before SIGKILL
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(2) # Give it time to react to SIGKILL
                process.wait(timeout=10) # Wait again after SIGKILL
                print(f"Process group {pgid} terminated via SIGKILL (exit code: {process.returncode}).", flush=True)
            except ProcessLookupError:
                 print(f"Process group {pgid} (PID: {process.pid}) already gone after SIGTERM timeout.", flush=True)
            except subprocess.TimeoutExpired:
                 print(f"Warning: Timeout waiting for process group {pgid} to terminate even after SIGKILL.", flush=True)
                 # Force kill the main process again if group kill seemed ineffective
                 if process.poll() is None:
                     try:
                         process.kill()
                         process.wait(timeout=5)
                     except Exception as final_kill_e:
                         print(f"Error during final process.kill() attempt: {final_kill_e}", flush=True)
            except Exception as e_kill:
                print(f"Error sending SIGKILL to process group {pgid}: {e_kill}", flush=True)
                traceback.print_exc()

        except ProcessLookupError:
            print(f"Process group {pgid} (PID: {process.pid}) already gone before SIGTERM.", flush=True)
        except Exception as e_term:
            print(f"Error sending SIGTERM to process group {pgid} (PID: {process.pid}): {e_term}", flush=True)
            traceback.print_exc()
            # Fallback kill attempt on the main process PID if group kill failed
            try:
                if process.poll() is None: process.terminate()
                time.sleep(2)
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
                print(f"Main process PID {process.pid} terminated (fallback).", flush=True)
            except Exception as fallback_e:
                 print(f"Error during fallback kill of PID {process.pid}: {fallback_e}", flush=True)

    elif process:
         exit_code_str = f"(exit code: {process.returncode})" if hasattr(process, 'returncode') and process.returncode is not None else ""
         print(f"Process (PID: {process.pid}) already terminated {exit_code_str} before kill attempt.", flush=True)

# --- Check Server Logs for Errors (Reads files, used during readiness) ---
def check_server_logs_for_errors(
    stdout_f: Optional[IO[Any]], # File handle opened in 'r' mode
    stderr_f: Optional[IO[Any]], # File handle opened in 'r' mode
    error_patterns: List[str],
    last_stdout_pos: int,
    last_stderr_pos: int,
) -> Tuple[bool, Optional[str], int, int]:
    """Reads new lines from server log files and checks for error patterns using regex."""
    new_stdout_pos = last_stdout_pos
    new_stderr_pos = last_stderr_pos
    found_pattern = None
    original_error_patterns = list(error_patterns) # Copy to avoid modifying caller's list

    files_to_check = []
    if stdout_f: files_to_check.append((stdout_f, last_stdout_pos, "stdout"))
    if stderr_f: files_to_check.append((stderr_f, last_stderr_pos, "stderr"))

    try:
        for f_handle, last_pos, log_type in files_to_check:
            if not f_handle or f_handle.closed: continue
            try:
                current_pos = f_handle.tell() # Get current position before reading
                f_handle.seek(last_pos)
                new_lines = f_handle.readlines() # Read only new lines since last check
                # Important: Reset position to end of file after reading new lines
                # This seems counter-intuitive, but seeking back is needed if the file
                # handle is shared or used elsewhere. Let's try just updating the pos var.
                # f_handle.seek(0, os.SEEK_END) # Seek to end? No, just update position variable
                current_pos = f_handle.tell() # Get new position *after* reading
            except ValueError: # Handle seeking on closed file
                 print(f"  Warning: Attempted to seek/read on closed log file handle for {log_type}", flush=True)
                 continue
            except Exception as read_err:
                 print(f"  Warning: Error reading {log_type} log file: {read_err}", flush=True)
                 continue # Skip checking this file on this iteration


            if log_type == "stdout": new_stdout_pos = current_pos
            else: new_stderr_pos = current_pos

            if not new_lines: continue

            for line in new_lines:
                if not line.strip(): continue
                for pattern in original_error_patterns:
                    try:
                        if re.search(pattern, line):
                            print(f"  ERROR DETECTED in {log_type}: Pattern '{pattern}' matched line: {line.strip()}", flush=True)
                            found_pattern = pattern
                            return True, found_pattern, new_stdout_pos, new_stderr_pos
                    except re.error as re_err:
                        print(f"  Warning: Invalid regex pattern '{pattern}': {re_err}", flush=True)
                        # Avoid repeated warnings by trying to remove from the original list if possible
                        # This might fail if error_patterns is used concurrently, safer to ignore remove error.
                        try: error_patterns.remove(pattern)
                        except ValueError: pass

    except IOError as e:
        print(f"  Warning: IOError check_server_logs_for_errors loop: {e}", flush=True)
        return False, None, new_stdout_pos, new_stderr_pos # Return current positions
    except Exception as e:
        print(f"  Warning: Unexpected error checking logs: {e}", flush=True)
        traceback.print_exc()
        return False, None, new_stdout_pos, new_stderr_pos # Return current positions

    return False, None, new_stdout_pos, new_stderr_pos


# --- Other Helper Functions (_get_model_family, generate_config_hash, completion markers, etc.) ---
def _get_model_family(model_id: str, template: Optional[str]) -> str:
    """Derives a model family name based on the template."""
    if not template: template = "{model_id_basename}"
    model_id_basename = model_id.split('/')[-1]
    try:
        return template.format(
            model_id=model_id,
            model_id_basename=model_id_basename
        )
    except KeyError as e:
        print(f"Warning: Invalid placeholder {e} in model.family_template. Using basename '{model_id_basename}'.", flush=True)
        return model_id_basename
    except Exception as e:
        print(f"Warning: Error formatting model.family_template: {e}. Using basename '{model_id_basename}'.", flush=True)
        return model_id_basename

def generate_config_hash(data: Any) -> str:
    """Generates an MD5 hash for a given Python object (dicts, lists, etc.)."""
    try:
        serialized_data = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(serialized_data.encode('utf-8')).hexdigest()
    except TypeError as e:
        print(f"Warning: Could not serialize data for hashing: {e}. Returning fallback hash.", flush=True)
        return hashlib.md5(str(data).encode('utf-8')).hexdigest()

def get_server_config_for_hashing(engine_name: str, tp: int, pp: int, config: Dict[str, Any]) -> Dict:
    """Extracts the relevant server configuration parts for hashing."""
    engine_def = config.get('engines', {}).get(engine_name, {})
    server_def = engine_def.get('server', {})
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
            "error_patterns": sorted(server_def.get('error_patterns', [])),
        },
        "engine_constraints": engine_def.get('constraints', {}),
        "runner_conda_base_env": config.get('runner',{}).get('conda_base_env'),
        "engine_conda_env_pattern": engine_def.get('environment', {}).get('conda_env_name'),
    }

def get_benchmark_config_for_hashing(
    benchmark_run_config: Dict[str, Any],
    engine_identifier: str,
    config: Dict[str, Any],
    resolved_bench_config_path: Optional[str]
) -> Dict:
    """Extracts the relevant benchmark configuration parts for hashing."""
    benchmark_type = benchmark_run_config['type']
    bench_def = config.get('benchmark_definitions', {}).get(benchmark_type, {})
    bench_env_pattern = bench_def.get('environment')
    bench_conda_env_name = None
    if bench_env_pattern:
         try:
             env_prefix = config.get('runner', {}).get('env_prefix', '')
             bench_conda_env_name = bench_env_pattern.format(env_prefix=env_prefix, benchmark_type=benchmark_type)
         except Exception: pass

    return {
        "benchmark_type": benchmark_type,
        "engine_identifier": engine_identifier,
        "model_config": config.get('model', {}),
        "benchmark_global_defaults": config.get('benchmark', {}).get('defaults', {}),
        "benchmark_definition": { # Hash relevant parts of definition
            "command_base": bench_def.get('command_base', []),
            "args_mapping": bench_def.get('args_mapping', {}),
            "defaults": bench_def.get('defaults', {}),
            # Don't hash environment pattern directly here, use resolved name below
        },
        "benchmark_run_overrides": benchmark_run_config.get('overrides', {}),
        "resolved_benchmark_config_path": resolved_bench_config_path,
        "benchmark_enabled_flag": benchmark_run_config.get('enabled', False),
        "runner_conda_base_env": config.get('runner',{}).get('conda_base_env'),
        "resolved_benchmark_conda_env": bench_conda_env_name # Hash the resolved name
    }

def check_completion_marker(marker_path: str, expected_server_hash: str, expected_benchmark_hash: str) -> bool:
    """Checks if a valid completion marker exists with matching hashes."""
    if not os.path.exists(marker_path):
        return False
    try:
        with open(marker_path, 'r') as f:
            marker_data = json.load(f)
        # Check if required keys and hashes match
        server_match = marker_data.get("server_config_hash") == expected_server_hash
        bench_match = marker_data.get("benchmark_config_hash") == expected_benchmark_hash

        if server_match and bench_match:
            # print(f"  Found valid completion marker: {marker_path}", flush=True) # Reduce noise
            return True
        else:
            reason = []
            if not server_match: reason.append("server hash mismatch")
            if not bench_match: reason.append("benchmark hash mismatch")
            print(f"  Found stale completion marker ({', '.join(reason)}): {marker_path}", flush=True)
            return False # Hashes don't match, needs re-run
    except (json.JSONDecodeError, IOError, KeyError, TypeError) as e:
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
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, 'w') as f:
            json.dump(marker_data, f, indent=2)
        print(f"  Wrote completion marker: {marker_path}", flush=True)
    except IOError as e:
        print(f"!!! Error writing completion marker {marker_path}: {e} !!!", flush=True)


# --- Command Generation Functions (get_server_command, get_benchmark_command) ---
def get_server_command(
    engine_name: str,
    tp: int,
    pp: int,
    port: int,
    config: Dict[str, Any]
) -> List[str]:
    """Builds the server start command based on the engine definition in config."""
    if engine_name not in config.get('engines', {}):
        raise ValueError(f"Engine '{engine_name}' not defined in the config file's 'engines' section.")

    engine_def = config['engines'][engine_name]
    server_def = engine_def.get('server', {})
    args_mapping = server_def.get('args_mapping', {})

    param_values = {
        'tp': tp,
        'pp': pp,
        'port': port,
        'model_id': config['model']['id'],
        'host': config['server']['host'],
    }
    merged_defaults = {**config['server'].get('defaults', {}), **server_def.get('defaults', {})}
    param_values = {**merged_defaults, **param_values}

    # --- Special Handling ---
    if engine_name == "vajra": # Example specific handling
        if param_values.get("scheduler") == "FIXED_CHUNK" and 'chunk_size' in args_mapping and 'chunk_size' in param_values:
            args_mapping['chunk_size'] = "--fixed_chunk_replica_scheduler_config_chunk_size {value}"
        if 'prioritizer' in param_values:
             param_values['prioritizer'] = str(param_values['prioritizer']).upper()
    # --- End Special Handling ---

    cmd = list(server_def.get('command_base', []))
    if not cmd:
        raise ValueError(f"Engine '{engine_name}' definition is missing 'server.command_base'.")

    positional_model_arg = None
    processed_keys = set()

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
                    # Use shlex.split to handle arguments with spaces if value contains them
                    formatted_arg = shlex.split(template.format(value=shlex.quote(str(value))))
                    cmd.extend(formatted_arg)
                except Exception as format_e:
                     print(f"Warning: Could not format arg for key '{key}' with template '{template}' and value '{value}': {format_e}", flush=True)
            elif isinstance(value, bool) and value: # Handle boolean flag (True case)
                cmd.append(template)
            elif isinstance(value, bool) and not value and template.startswith("--no-"): # Handle boolean flag (False case with --no- prefix)
                 cmd.append(template)
            elif not isinstance(value, bool) and "{value}" not in template: # Allow non-boolean flags without {value} placeholder
                 cmd.append(template)
            # else: ignore non-boolean value if no placeholder, or boolean False for regular flag

    # Handle positional model argument
    if server_def.get('server_model_arg_is_positional', False):
         if 'model_id' in param_values:
             positional_model_arg = str(param_values['model_id'])
             processed_keys.add('model_id')
         else:
             print(f"Warning: Engine '{engine_name}' expects positional model_id, but 'model_id' not found.", flush=True)

    if positional_model_arg:
         cmd.append(positional_model_arg)

    return cmd

def get_benchmark_command(
    benchmark_config: Dict[str, Any],
    engine_identifier: str,
    host: str,
    port: int,
    run_base_dir: str,
    config: Dict[str, Any],
    main_config_path: str
) -> Tuple[List[str], str, str, Optional[str], Optional[str]]:
    """Builds a benchmark command. Returns: command_list, log_dir, log_prefix, conda_env_name, resolved_bench_config_path."""
    benchmark_type = benchmark_config['type']
    if benchmark_type not in config.get('benchmark_definitions', {}):
        raise ValueError(f"Benchmark type '{benchmark_type}' not defined in benchmark_definitions.")

    bench_def = config['benchmark_definitions'][benchmark_type]
    args_mapping = bench_def.get('args_mapping', {})

    # Define log dir and prefix first
    bench_log_dir = os.path.join(run_base_dir, f"benchmark_{benchmark_type}")
    log_prefix = f"{benchmark_type}_benchmark"

    # Gather parameter values
    param_values = {
        'host': host,
        'port': port,
        'model_id': config['model']['id'],
        'engine_identifier': engine_identifier,
        'output_dir': bench_log_dir, # Pass the specific dir
        'api_base': f"http://{host}:{port}/v1", # Default
    }
    merged_defaults = {
        **config['benchmark'].get('defaults', {}),
        **bench_def.get('defaults', {}),
        **benchmark_config.get('overrides', {})
    }
    param_values = {**merged_defaults, **param_values}

    # --- Special Handling: Benchmark Internal Config Path ---
    resolved_bench_config_path: Optional[str] = None
    if 'config_path_template' in param_values and 'config_path' in args_mapping:
        template = param_values['config_path_template']
        model_family = _get_model_family(config['model']['id'], config['model'].get('family_template'))
        config_dir = os.path.dirname(main_config_path) if main_config_path else "."
        try:
            resolved_path_str = template.format(
                model_family=model_family, model_id=config['model']['id'], config_dir=config_dir
            )
            resolved_path = Path(config_dir) / resolved_path_str
            resolved_bench_config_path = str(resolved_path.resolve())
            param_values['config_path'] = resolved_bench_config_path # Update param for command gen
            # print(f"  Resolved benchmark config path: {param_values['config_path']}", flush=True) # Reduce noise
        except KeyError as e:
            print(f"Warning: Invalid placeholder {e} in benchmark config_path_template '{template}'. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None)
        except Exception as e:
            print(f"Warning: Error resolving benchmark config_path_template '{template}': {e}. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None)
    elif 'config_path' in param_values: # Handle directly provided path
        config_dir = os.path.dirname(main_config_path) if main_config_path else "."
        potential_path = Path(config_dir) / param_values['config_path']
        if potential_path.is_absolute():
             resolved_bench_config_path = param_values['config_path']
        elif potential_path.exists():
            resolved_bench_config_path = str(potential_path.resolve())
            param_values['config_path'] = resolved_bench_config_path # Update param
        else:
            resolved_bench_config_path = param_values['config_path'] # Keep as is if not found relative
            # print(f"  Note: Benchmark config_path '{resolved_bench_config_path}' not found relative to main config, using as is.", flush=True) # Reduce noise
    # --- End Special Handling ---

    # Build command list
    cmd = list(bench_def.get('command_base', []))
    if not cmd:
        raise ValueError(f"Benchmark definition '{benchmark_type}' is missing 'command_base'.")

    processed_keys = set()
    for key, template in sorted(args_mapping.items()):
        if key in param_values and param_values[key] is not None:
            value = param_values[key]
            processed_keys.add(key)
            if template == "{value}": # Special case: just the value
                cmd.append(str(value))
            elif "{value...}" in template: # List expansion
                 if isinstance(value, list):
                    base_arg = template.split('{value...}')[0].strip()
                    if base_arg: cmd.append(base_arg)
                    cmd.extend(map(str, value))
                 else:
                    print(f"Warning: Arg mapping for '{key}' expects list ('{{value...}}') but got {type(value)}. Ignoring.", flush=True)
            elif "{value}" in template: # Standard substitution
                try:
                    formatted_arg = shlex.split(template.format(value=shlex.quote(str(value))))
                    cmd.extend(formatted_arg)
                except Exception as format_e:
                     print(f"Warning: Could not format benchmark arg for key '{key}' with template '{template}' and value '{value}': {format_e}", flush=True)
            elif isinstance(value, bool): # Boolean flags
                 if value is True: cmd.append(template)
                 elif value is False and template.startswith("--no-"): cmd.append(template)
            elif not isinstance(value, bool) and "{value}" not in template: # Flag based on key presence
                 cmd.append(template)

    # Determine conda environment
    bench_env_pattern = bench_def.get('environment')
    bench_conda_env_name = None
    if bench_env_pattern:
         try:
             env_prefix = config.get('runner', {}).get('env_prefix', '')
             bench_conda_env_name = bench_env_pattern.format(env_prefix=env_prefix, benchmark_type=benchmark_type)
         except KeyError as e: print(f"Warning: Invalid placeholder {e} in benchmark environment pattern for {benchmark_type}.", flush=True)
         except Exception as e: print(f"Warning: Error formatting benchmark environment pattern '{bench_env_pattern}': {e}.", flush=True)

    return cmd, bench_log_dir, log_prefix, bench_conda_env_name, resolved_bench_config_path

# --- Config Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file and performs validation."""
    print(f"Loading configuration from: {config_path}", flush=True)
    abs_config_path = str(Path(config_path).resolve())
    try:
        with open(abs_config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict): raise ValueError("Config file must contain a YAML dictionary.")
        print("Configuration loaded successfully.", flush=True)
        required_keys = ['runner', 'model', 'server', 'benchmark', 'engines', 'run_matrix', 'benchmark_definitions']
        for key in required_keys:
            if key not in config: raise ValueError(f"Missing required top-level key in config: '{key}'")
        if not isinstance(config.get('benchmark', {}).get('run'), list): raise ValueError("'benchmark.run' must be a list.")
        if not isinstance(config.get('benchmark_definitions'), dict): raise ValueError("'benchmark_definitions' must be a dict.")

        defined_benchmarks = set(config['benchmark_definitions'].keys())
        for run_conf in config['benchmark']['run']:
            if not isinstance(run_conf, dict) or 'type' not in run_conf: raise ValueError("Each item in 'benchmark.run' must be a dict with 'type'.")
            if run_conf['type'] not in defined_benchmarks: raise ValueError(f"Benchmark type '{run_conf['type']}' in 'benchmark.run' not defined.")

        defined_engines = set(config.get('engines', {}).keys())
        for engine_to_run in config['run_matrix'].get('engines', []):
            if engine_to_run not in defined_engines: raise ValueError(f"Engine '{engine_to_run}' in 'run_matrix' not defined.")

        for engine_name, engine_data in config.get('engines', {}).items():
            error_patterns = engine_data.get('server', {}).get('error_patterns')
            if error_patterns is not None:
                if not isinstance(error_patterns, list): raise ValueError(f"Engine '{engine_name}' server.error_patterns must be a list.")
                if not all(isinstance(item, str) for item in error_patterns): raise ValueError(f"Engine '{engine_name}' server.error_patterns must contain only strings.")

        config['_main_config_path'] = abs_config_path
        return config
    except FileNotFoundError: print(f"!!! Error: Config file not found at {config_path} (resolved to {abs_config_path}) !!!", flush=True); raise
    except yaml.YAMLError as e: print(f"!!! Error parsing YAML config file {abs_config_path}: {e} !!!", flush=True); raise
    except ValueError as e: print(f"!!! Config file validation error: {e} !!!", flush=True); raise
    except Exception as e: print(f"!!! An unexpected error occurred loading config: {e} !!!", flush=True); traceback.print_exc(); raise


# --- Main Execution Logic (With Pre-Check Optimization) ---
def main():
    parser = argparse.ArgumentParser(description="Run LLM Engine Benchmarks using Pluggable YAML Config Definitions")
    parser.add_argument("--config", type=str, default="./scripts/benchmark_config.yml", help="Path to the YAML configuration file.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore completion markers and force all benchmarks to run.")
    parser.add_argument("--no-console-status", action="store_true", help="Suppress status messages from the main script to the console.")
    args = parser.parse_args()

    # Determine context for redirection
    output_context = contextlib.nullcontext()
    if args.no_console_status:
        print("Note: --no-console-status activated. Main script output will be suppressed.", file=sys.__stderr__)
        output_context = redirect_stdout_stderr()

    # Run main logic within the (potentially redirecting) context
    with output_context:
        try:
            config = load_config(args.config)
        except Exception:
            if args.no_console_status: traceback.print_exc(file=sys.__stderr__)
            exit(1)

        main_config_path = config['_main_config_path']
        runner_config = config['runner']
        server_config = config['server']
        benchmark_control = config['benchmark']
        run_matrix = config['run_matrix']
        engine_definitions = config['engines']
        base_output_dir = runner_config['base_output_dir']
        os.makedirs(base_output_dir, exist_ok=True)
        current_port = server_config['start_port']
        parallel_combinations = list(itertools.product(run_matrix.get('tp_dims', [1]), run_matrix.get('pp_dims', [1])))
        env_prefix = runner_config.get('env_prefix', '')
        stream_all_logs = runner_config.get('stream_logs', False)
        if args.no_console_status: stream_all_logs = False # Force silence subprocess console

        # --- Main Loop ---
        for engine_name in run_matrix.get('engines', []):
            engine_def = engine_definitions.get(engine_name)
            if not engine_def: continue
            engine_constraints = engine_def.get('constraints', {})

            for tp, pp in parallel_combinations:
                # --- Constraint Checks (Before Pre-check) ---
                gpus_required = tp * pp
                max_gpus = runner_config.get('max_gpus', 999)
                if gpus_required > max_gpus:
                    print(f"\n--- Skipping {engine_name} TP={tp}, PP={pp} (requires {gpus_required} GPUs > max {max_gpus}) ---\n", flush=True)
                    current_port += 1 # Ensure port increments even if skipped here
                    continue
                max_pp_allowed = engine_constraints.get('max_pp')
                if max_pp_allowed is not None and pp > max_pp_allowed:
                     print(f"\n--- Skipping {engine_name} TP={tp}, PP={pp} (PP > max {max_pp_allowed} for this engine) ---\n", flush=True)
                     current_port += 1 # Ensure port increments even if skipped here
                     continue

                engine_tp_pp_id = f"{engine_name}_tp{tp}_pp{pp}"
                run_base_dir = os.path.join(base_output_dir, engine_tp_pp_id)
                host_for_run = server_config['host'] # Use configured host
                port_for_run = current_port # Use the current port number

                # --- *** Optimization: Pre-check Benchmark Completion *** ---
                server_config_data = get_server_config_for_hashing(engine_name, tp, pp, config)
                server_config_hash = generate_config_hash(server_config_data)
                all_benchmarks_complete = True # Assume true initially
                benchmarks_to_run_exist = False # Track if any enabled benchmarks exist for this combo

                if not args.force_rerun:
                    print(f"\n--- Pre-checking benchmark completion for {engine_tp_pp_id} ---", flush=True)
                    for bench_run_config in benchmark_control.get('run', []):
                        if not bench_run_config.get('enabled', False):
                            continue # Skip disabled benchmarks

                        benchmarks_to_run_exist = True # Found at least one enabled benchmark
                        benchmark_type = bench_run_config['type']
                        print(f"  Checking: {benchmark_type}...", end='', flush=True)

                        try:
                             # Need to simulate parts of benchmark setup to get paths/hashes
                            _, bench_log_dir, _, _, resolved_bench_config_path = get_benchmark_command(
                                benchmark_config=bench_run_config, engine_identifier=engine_tp_pp_id,
                                host=host_for_run, port=port_for_run, run_base_dir=run_base_dir,
                                config=config, main_config_path=main_config_path
                            )
                            completion_marker_path = os.path.join(bench_log_dir, COMPLETION_MARKER_FILENAME)
                            bench_config_data = get_benchmark_config_for_hashing(
                                bench_run_config, engine_tp_pp_id, config, resolved_bench_config_path
                            )
                            benchmark_config_hash = generate_config_hash(bench_config_data)

                            # Check the marker
                            if not check_completion_marker(completion_marker_path, server_config_hash, benchmark_config_hash):
                                print(" incomplete.", flush=True)
                                all_benchmarks_complete = False
                                break # Found an incomplete one, no need to check further
                            else:
                                print(" complete.", flush=True)

                        except Exception as precheck_e:
                             # If any error occurs during pre-check (e.g., getting command/hash), assume incomplete
                             print(f" error checking ({type(precheck_e).__name__}), assuming incomplete.", flush=True)
                             all_benchmarks_complete = False
                             break

                    if benchmarks_to_run_exist and all_benchmarks_complete:
                        print(f"--- Skipping server launch for {engine_tp_pp_id}: All enabled benchmarks already completed. ---\n", flush=True)
                        current_port += 1 # Increment port for the next potential server
                        continue # Skip to the next tp/pp combination
                    elif not benchmarks_to_run_exist:
                         print(f"--- Skipping server launch for {engine_tp_pp_id}: No benchmarks enabled for this configuration. ---\n", flush=True)
                         current_port += 1
                         continue
                    else:
                         print(f"--- Proceeding with server launch for {engine_tp_pp_id}: At least one benchmark needs to run. ---", flush=True)
                else:
                     print(f"--- Proceeding with server launch for {engine_tp_pp_id} (--force-rerun specified). ---", flush=True)
                # --- *** End of Pre-check Optimization *** ---


                # --- Server Launch and Benchmark Execution (Only if not skipped) ---
                print(f"\n{'='*25} Starting Run: {engine_tp_pp_id} {'='*25}\n", flush=True)
                server_process: Optional[subprocess.Popen] = None
                server_stdout_thread: Optional[threading.Thread] = None
                server_stderr_thread: Optional[threading.Thread] = None
                detected_fatal_error = False
                server_ready_status = False
                server_stdout_log = ""
                server_stderr_log = ""
                server_launch_successful = False # Track if server section completes to manage final cleanup msg

                try:
                    # 1. Setup Server Logging Dir & Get Paths
                    server_log_dir = os.path.join(run_base_dir, "server_logs")
                    server_stdout_log, server_stderr_log = setup_logging(server_log_dir, "server")

                    # 2. Determine Server Conda Env
                    server_env_pattern = engine_def.get('environment', {}).get('conda_env_name')
                    server_conda_env_name = None
                    if server_env_pattern:
                        try: server_conda_env_name = server_env_pattern.format(env_prefix=env_prefix, engine_name=engine_name)
                        except Exception as e: print(f"Warning: Failed to format server conda env name for {engine_name}: {e}", flush=True)

                    # 3. Clean up temp dirs
                    if runner_config.get('cleanup_tmp', True):
                        print(f"--- Cleaning up temporary LLM workspace directories ---", flush=True)
                        cleanup_proc, cleanup_stdout_t, cleanup_stderr_t = run_command(
                            ["rm", "-rf", "/tmp/*-llm-workspace"], check_return_code=False, stream_logs=stream_all_logs
                        )
                        if cleanup_proc: cleanup_proc.wait(timeout=10)
                        if cleanup_stdout_t: cleanup_stdout_t.join(timeout=1)
                        if cleanup_stderr_t: cleanup_stderr_t.join(timeout=1)

                    # 4. Construct & Start Server Command (Asynchronously)
                    print(f"--- Launching {engine_name} Server (TP={tp}, PP={pp}) ---", flush=True)
                    server_cmd_list = get_server_command(engine_name=engine_name, tp=tp, pp=pp, port=port_for_run, config=config)

                    server_process, server_stdout_thread, server_stderr_thread = run_command(
                        server_cmd_list, env_name=server_conda_env_name, conda_base_env=runner_config.get('conda_base_env'),
                        stdout_log_path=server_stdout_log, stderr_log_path=server_stderr_log,
                        stream_logs=stream_all_logs, check_return_code=False
                    )

                    if server_process is None: raise RuntimeError(f"Failed to start server process for {engine_tp_pp_id}.")
                    server_launch_successful = True # Mark that we got past Popen
                    time.sleep(3) # Allow process start

                    # 5. Combined Readiness & Error Check Loop
                    print(f"--- Waiting for server readiness & monitoring logs ({server_config['readiness_timeout']}s timeout) ---", flush=True)
                    start_time = time.monotonic()
                    error_patterns = engine_def.get('server', {}).get('error_patterns', [])
                    last_stdout_pos, last_stderr_pos = 0, 0
                    read_stdout_f, read_stderr_f = None, None

                    try:
                        # Open log files for *reading* by the error checker
                        if server_stdout_log and os.path.exists(server_stdout_log): read_stdout_f = open(server_stdout_log, 'r', encoding='utf-8', errors='replace')
                        if server_stderr_log and os.path.exists(server_stderr_log): read_stderr_f = open(server_stderr_log, 'r', encoding='utf-8', errors='replace')

                        while time.monotonic() - start_time < server_config['readiness_timeout']:
                            if server_process.poll() is not None: # Check 1: Process died?
                                if server_stdout_thread: server_stdout_thread.join(timeout=5) # Ensure logs flushed
                                if server_stderr_thread: server_stderr_thread.join(timeout=5)
                                raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) exited prematurely (code {server_process.returncode}). Logs: {server_log_dir}")

                            if error_patterns and (read_stdout_f or read_stderr_f): # Check 2: Errors in logs?
                                found_error, error_msg, last_stdout_pos, last_stderr_pos = check_server_logs_for_errors(
                                    read_stdout_f, read_stderr_f, error_patterns, last_stdout_pos, last_stderr_pos
                                )
                                if found_error:
                                    detected_fatal_error = True
                                    print(f"\n!!! Detected fatal server error pattern: '{error_msg}' in logs for {engine_tp_pp_id} !!!", flush=True)
                                    break # Exit readiness loop

                            # Check 3: API Ready?
                            api_check_url = f"http://{host_for_run}:{port_for_run}/v1/models" # Example endpoint
                            try:
                                response = requests.get(api_check_url, timeout=2)
                                if response.status_code == 200:
                                    print(f"\nServer API is ready! ({api_check_url} responded {response.status_code} in {time.monotonic() - start_time:.2f}s)", flush=True)
                                    server_ready_status = True
                                    break
                                else: print(f"S({response.status_code})", end='', flush=True)
                            except ConnectionError: print(".", end='', flush=True)
                            except Timeout: print("T", end='', flush=True)
                            except Exception as api_e: print(f"E({type(api_e).__name__})", end='', flush=True)

                            time.sleep(2)
                    finally:
                        if read_stdout_f: read_stdout_f.close()
                        if read_stderr_f: read_stderr_f.close()
                    # --- End of Readiness & Error Check Loop ---

                    # 6. Post-Loop Handling & Benchmark Execution
                    if detected_fatal_error:
                        print(f"--- Skipping benchmarks for {engine_tp_pp_id} due to detected server error during startup. ---", flush=True)
                    elif not server_ready_status:
                         server_exit_code = server_process.poll()
                         if server_stdout_thread: server_stdout_thread.join(timeout=5)
                         if server_stderr_thread: server_stderr_thread.join(timeout=5)
                         if server_exit_code is None: raise RuntimeError(f"{engine_name} server did not become ready within {server_config['readiness_timeout']}s. Server might be hung.")
                         else: raise RuntimeError(f"{engine_name} server exited (code {server_exit_code}) during timeout period without becoming ready.")
                    else:
                        # --- Server Ready - Benchmark Execution Loop ---
                        print("--- Server is running and ready, proceeding to benchmarks ---", flush=True)
                        for bench_run_config in benchmark_control.get('run', []):
                            if not bench_run_config.get('enabled', False): continue # Skip disabled

                            benchmark_type = bench_run_config['type']
                            print(f"\n--- Preparing Benchmark: {benchmark_type} for {engine_tp_pp_id} ---", flush=True)

                            bench_proc: Optional[subprocess.Popen] = None
                            bench_stdout_t: Optional[threading.Thread] = None
                            bench_stderr_t: Optional[threading.Thread] = None

                            try:
                                # Get command, paths, env, resolved config path
                                bench_cmd, bench_log_dir, bench_log_prefix, bench_conda_env, resolved_bench_config_path = get_benchmark_command(
                                    benchmark_config=bench_run_config, engine_identifier=engine_tp_pp_id,
                                    host=host_for_run, port=port_for_run, run_base_dir=run_base_dir,
                                    config=config, main_config_path=main_config_path
                                )

                                # Calculate Benchmark Config Hash (using current server hash)
                                bench_config_data = get_benchmark_config_for_hashing(
                                    bench_run_config, engine_tp_pp_id, config, resolved_bench_config_path
                                )
                                benchmark_config_hash = generate_config_hash(bench_config_data)
                                print(f"  Benchmark Config Hash: {benchmark_config_hash}", flush=True)

                                # Define completion marker path
                                completion_marker_path = os.path.join(bench_log_dir, COMPLETION_MARKER_FILENAME)

                                # Check for Completion Marker (again, respect force-rerun)
                                if not args.force_rerun:
                                    if check_completion_marker(completion_marker_path, server_config_hash, benchmark_config_hash):
                                        print(f"--- Skipping Benchmark: {benchmark_type} for {engine_tp_pp_id} (already completed with matching config) ---", flush=True)
                                        continue
                                else:
                                    print("  --force-rerun specified, ignoring completion marker check.", flush=True)

                                # Setup benchmark logging paths
                                bench_stdout_log, bench_stderr_log = setup_logging(bench_log_dir, bench_log_prefix)
                                final_conda_env = bench_conda_env if bench_conda_env is not None else server_conda_env_name

                                # Execute benchmark command (Wait for it to finish)
                                print(f"--- Running Benchmark: {benchmark_type} for {engine_tp_pp_id} ---", flush=True)
                                try:
                                    bench_proc, bench_stdout_t, bench_stderr_t = run_command(
                                        bench_cmd, env_name=final_conda_env, conda_base_env=runner_config.get('conda_base_env'),
                                        stdout_log_path=bench_stdout_log, stderr_log_path=bench_stderr_log,
                                        stream_logs=stream_all_logs, check_return_code=True # Wait and check code
                                    )
                                    print(f"--- Benchmark {benchmark_type} completed successfully ---", flush=True)
                                    write_completion_marker(completion_marker_path, server_config_hash, benchmark_config_hash)

                                except subprocess.CalledProcessError as bench_err:
                                    rc = bench_err.returncode
                                    print(f"!!! Benchmark '{benchmark_type}' failed for {engine_tp_pp_id} (exit code: {rc}) !!!", flush=True)
                                except Exception as bench_exec_err:
                                    print(f"!!! Benchmark '{benchmark_type}' failed to execute for {engine_tp_pp_id}: {bench_exec_err} !!!", flush=True)

                            except Exception as bench_setup_e:
                                print(f"!!! An error occurred during benchmark '{benchmark_type}' setup for {engine_tp_pp_id}: {bench_setup_e} !!!", flush=True)
                                traceback.print_exc()

                            finally:
                                # Benchmark threads joined by run_command(check=True), process should be finished or error raised
                                pass # No specific cleanup needed here usually

                except Exception as run_e:
                    # Catch errors during server launch, readiness, or benchmark setup phase
                    print(f"\n!!! === ERROR during run for {engine_tp_pp_id}: {run_e} === !!!\n", flush=True)
                    if not isinstance(run_e, (subprocess.CalledProcessError, FileNotFoundError, RuntimeError)):
                         traceback.print_exc()
                    if args.no_console_status: print(f"CRITICAL ERROR in {engine_tp_pp_id}: {run_e}", file=sys.__stderr__)


                finally:
                    # --- Cleanup Server and Log Threads ---
                    # Only perform server cleanup if launch was attempted/successful
                    if server_launch_successful:
                        print(f"\n--- Cleaning up server and log threads for {engine_tp_pp_id} ---", flush=True)
                        kill_process_group(server_process) # Kill the server process group

                        # Wait for the server log reader threads to finish
                        if server_stdout_thread and server_stdout_thread.is_alive():
                            server_stdout_thread.join(timeout=10)
                            if server_stdout_thread.is_alive(): print("Warning: Server stdout log thread join timed out.", flush=True)
                        if server_stderr_thread and server_stderr_thread.is_alive():
                            server_stderr_thread.join(timeout=10)
                            if server_stderr_thread.is_alive(): print("Warning: Server stderr log thread join timed out.", flush=True)

                    # Reset state variables for the next loop iteration
                    server_process, server_stdout_thread, server_stderr_thread = None, None, None
                    detected_fatal_error = False
                    server_ready_status = False
                    server_launch_successful = False

                    print(f"\n{'='*25} Finished Run/Skip: {engine_tp_pp_id} {'='*25}\n", flush=True)
                    # Increment port AFTER the loop for this tp/pp combo completes or is skipped
                    current_port += 1
                    time.sleep(runner_config.get('sleep_between_runs', 5)) # Use configured sleep time

        print("\nAll configured benchmark runs completed, skipped, or failed.", flush=True)
        # --- End of code block affected by output_context ---

if __name__ == "__main__":
    main()
