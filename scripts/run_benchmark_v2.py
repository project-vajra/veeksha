import subprocess
import time
import os
import signal
import shlex
import argparse
import itertools
import yaml
import requests
import re # For model family extraction
from pathlib import Path # For easier path manipulation
from requests.exceptions import ConnectionError, Timeout
from typing import List, Tuple, Dict, Optional, Any, IO

# --- Helper Functions (setup_logging, run_command, kill_process_group_and_close_logs, wait_for_server_ready - Ensure run_command takes conda_base_env) ---
# ... (Include the helper functions from the previous version, ensure run_command is up-to-date) ...
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
    check: bool = True,
    stdout_log_path: Optional[str] = None,
    stderr_log_path: Optional[str] = None,
    stream_logs: bool = False,
) -> Optional[Tuple[subprocess.Popen, Optional[IO[Any]], Optional[IO[Any]]]]: # Type hint IO
    """
    Runs a command, optionally within a conda environment, prints it,
    handles execution, and manages logging.
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
                conda_base = f"/home/azrsadmin/miniforge3/envs/{env_name}"
                print(f"  Warning: Cannot auto-detect conda base. Set 'conda_base_env' in config or ensure conda is in PATH.", flush=True)

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
        elif env_name: # Fallback to name if path check fails or no base found
             print(f"  Warning: Conda env path for '{env_name}' not found or base missing, attempting activation by name.", flush=True)
             full_cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd_list
        else: # Should not happen if env_name is set, but as safeguard
             full_cmd = cmd_list # Run directly if env specified but unusable
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
            return process, stdout_f, stderr_f
        else:
            result = subprocess.run(
                full_cmd,
                check=False,
                text=True,
                capture_output=True, # Changed from stream=True for better log handling
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
                raise subprocess.CalledProcessError(result.returncode, full_cmd, output=result.stdout, stderr=result.stderr)
            return None

    except subprocess.CalledProcessError as e:
        print(f"!!! Command failed with exit code {e.returncode} !!! Logs are in the files above.", flush=True)
        if check: raise
        return None
    except FileNotFoundError as e:
        print(f"!!! Command or Conda environment not found: {e}. Is conda installed/in PATH? Is env '{env_name}' correct? Check 'conda_base_env' or ensure conda command works. !!!", flush=True)
        if popen and not check: return None
        raise
    except Exception as e:
        print(f"!!! An unexpected error occurred while running command: {e} !!!", flush=True)
        if popen and not check: return None
        raise
    finally:
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

    # Close log files safely
    for f_handle in [stdout_log_f, stderr_log_f]:
        if f_handle and not f_handle.closed:
            try:
                f_handle.close()
            except Exception as e:
                print(f"Error closing log file: {e}", flush=True)

def wait_for_server_ready(host: str, port: int, timeout: int, check_endpoint: str = "/v1/models") -> bool:
    """Polls the server API endpoint until it's ready or timeout occurs."""
    start_time = time.monotonic()
    # Default check endpoint might need overriding for non-openai servers
    # For now, assume OpenAI compatible for readiness check
    url = f"http://{host}:{port}{check_endpoint}"
    print(f"Waiting for server at {url} to be ready (timeout: {timeout}s)...", flush=True)
    while time.monotonic() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5) # Short timeout for individual check
            if response.status_code == 200:
                print(f"Server is ready! (responded in {time.monotonic() - start_time:.2f}s)", flush=True)
                return True
            else:
                print(f"S({response.status_code})", end='', flush=True)
        except ConnectionError:
            print(".", end='', flush=True)
        except Timeout:
            print("T", end='', flush=True)
        except Exception as e:
            print(f"E({type(e).__name__})", end='', flush=True)

        if time.monotonic() - start_time >= timeout:
            break
        time.sleep(2)

    print(f"\nServer readiness check failed after {timeout} seconds.", flush=True)
    return False

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


# --- Command Generation Functions ---

def get_server_command(
    engine_name: str,
    tp: int,
    pp: int,
    port: int,
    model_run_dir: str, # Keep for context/potential future use
    config: Dict[str, Any]
) -> List[str]:
    """Builds the server start command based on the engine definition in config."""
    # (Identical to the previous version - Version 2)
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
    merged_defaults = {**config['server'].get('defaults', {}), **server_def.get('defaults', {})}
    param_values = {**merged_defaults, **param_values}

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
    for key, template in args_mapping.items():
        if key in param_values and param_values[key] is not None:
            value = param_values[key]
            processed_keys.add(key)

            if "{value...}" in template: # Handle list expansion
                if isinstance(value, list):
                    base_arg = template.replace("{value...}", "").strip()
                    cmd.append(base_arg)
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
                 # This assumes the template *is* the negative flag, e.g., "--no-feature"
                 # This logic might need refinement based on conventions
                 cmd.append(template)
            elif not isinstance(value, bool): # Non-boolean without {value} template
                 print(f"Warning: Argument template '{template}' for key '{key}' has no '{{value}}' placeholder but value ('{value}') is not boolean True. Ignoring.", flush=True)

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

    # Optional: Warn about unused parameters?
    unused_params = set(param_values.keys()) - processed_keys - {'scheduler'} # Example exclusion
    if unused_params:
        # print(f"Note: The following server parameters were available but not used by mappings for {engine_name}: {unused_params}", flush=True)
        pass # Reduce noise

    return cmd


def get_benchmark_command(
    benchmark_config: Dict[str, Any], # The specific entry from benchmark.run list
    engine_identifier: str, # e.g., "sglang_tp1_pp1"
    host: str,
    port: int,
    run_base_dir: str, # Base directory for the engine/tp/pp run
    config: Dict[str, Any], # Full config
    main_config_path: str # Path to the loaded config file
) -> Tuple[List[str], str, str, Optional[str]]: # Returns command, log_dir, log_prefix, conda_env_name
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
            param_values['config_path'] = str(resolved_path.resolve()) # Use absolute path
            print(f"  Resolved benchmark config path: {param_values['config_path']}", flush=True)
        except KeyError as e:
            print(f"Warning: Invalid placeholder {e} in benchmark config_path_template '{template}'. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None) # Remove if resolution failed
        except Exception as e:
            print(f"Warning: Error resolving benchmark config_path_template '{template}': {e}. Cannot set config_path.", flush=True)
            param_values.pop('config_path', None)

    # --- End Special Handling ---


    # 3. Build the command list
    cmd = list(bench_def.get('command_base', []))
    if not cmd:
        raise ValueError(f"Benchmark definition '{benchmark_type}' is missing 'command_base'.")

    # 4. Append arguments based on mapping
    processed_keys = set()
    for key, template in args_mapping.items():
        if key in param_values and param_values[key] is not None:
            value = param_values[key]
            processed_keys.add(key) # Mark key as used by mapping

            # --- Handle different template types ---
            if template == "{value}": # Special case: just the value (e.g. positional arg) - unlikely for benchmarks?
                cmd.append(str(value))
            elif "{value...}" in template: # List expansion
                 if isinstance(value, list):
                    # Get flag part, e.g. "--lengths" from "--lengths {value...}"
                    base_arg = template.split('{value...}')[0].strip()
                    if base_arg: # Only add flag if it exists
                        cmd.append(base_arg)
                    cmd.extend(map(str, value)) # Add list items
                 else:
                    print(f"Warning: Arg mapping for '{key}' expects list ('{{value...}}') but got {type(value)}. Ignoring.", flush=True)

            elif "{value}" in template: # Standard substitution "--flag {value}" or "--flag={value}"
                try:
                    # Use shlex.split to handle spaces correctly, e.g. "--flag value" vs "--flag=value"
                    formatted_arg = shlex.split(template.format(value=str(value)))
                    cmd.extend(formatted_arg)
                except Exception as format_e:
                     print(f"Warning: Could not format benchmark arg for key '{key}' with template '{template}' and value '{value}': {format_e}", flush=True)

            elif isinstance(value, bool): # Boolean flags
                 if value is True:
                     # Template is the flag itself, e.g., "--enable-feature"
                     cmd.append(template)
                 elif value is False and template.startswith("--no-"):
                      # Template is the negative flag, e.g., "--no-feature"
                      cmd.append(template)
                 # If value is False and template doesn't start with --no-, we assume the flag should be omitted.
            elif key == 'use_output_dir_flag' and template: # Handle simple presence flag explicitly
                 # This key signals a flag that should just be added if defined in mapping
                 cmd.append(template)

            # Note: This doesn't explicitly handle boolean 'false' for flags *not* starting with '--no-'.
            # Add specific logic if needed for benchmarks with different conventions.

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


    # Optional: Warn about unused parameters
    # internal_params = {'config_path_template'} # Params used internally by script logic
    # unused_params = set(param_values.keys()) - processed_keys - internal_params
    # if unused_params:
    #     print(f"Note: The following benchmark parameters were available but not used by mappings for {benchmark_type}: {unused_params}", flush=True)

    return cmd, bench_log_dir, log_prefix, bench_conda_env_name


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
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


# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description="Run LLM Engine Benchmarks using Pluggable YAML Config Definitions")
    parser.add_argument("--config", type=str, default="scripts/bechmark_config.yml", help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
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


            print(f"\n{'='*25} Starting Run: {engine_tp_pp_id} {'='*25}\n", flush=True)
            server_process = None
            server_stdout_f, server_stderr_f = None, None

            try:
                # 1. Setup Server Logging Dir (Logs will go inside)
                server_log_dir = os.path.join(run_base_dir, "server_logs")
                server_stdout_log, server_stderr_log = setup_logging(server_log_dir, "server")

                # 2. Clean up temp dirs
                if runner_config.get('cleanup_tmp', True):
                    print(f"--- Cleaning up temporary LLM workspace directories ---", flush=True)
                    subprocess.run("rm -rf /tmp/*-llm-workspace", shell=True, check=False)

                # 3. Construct & Start Server Command
                print(f"--- Launching {engine_name} Server (TP={tp}, PP={pp}) ---", flush=True)
                server_cmd_list = get_server_command(
                    engine_name=engine_name, tp=tp, pp=pp,
                    port=current_port, model_run_dir=run_base_dir, config=config
                )
                server_run_result = run_command(
                    server_cmd_list,
                    env_name=server_conda_env_name,
                    conda_base_env=runner_config.get('conda_base_env'),
                    popen=True, check=False,
                    stdout_log_path=server_stdout_log, stderr_log_path=server_stderr_log
                )

                if server_run_result is None:
                    raise RuntimeError(f"Failed to start server process for {engine_tp_pp_id}. Check logs or conda setup ({server_conda_env_name}).")
                server_process, server_stdout_f, server_stderr_f = server_run_result

                time.sleep(2) # Increased sleep slightly
                if server_process.poll() is not None:
                    raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) failed immediately on launch (code {server_process.returncode}). Logs: {server_log_dir}")

                # 4. Wait for Server Readiness
                server_host = server_config['host']
                if not wait_for_server_ready(server_host, current_port, server_config['readiness_timeout']):
                     if server_process.poll() is not None:
                         raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) exited prematurely (code {server_process.returncode}) during readiness check. Logs: {server_log_dir}")
                     else:
                        raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) did not become ready within {server_config['readiness_timeout']}s. Server might be hanging/slow. Logs: {server_log_dir}")

                if server_process.poll() is not None:
                    raise RuntimeError(f"{engine_name} server (PID: {server_process.pid}) died unexpectedly (code {server_process.returncode}) just after becoming ready. Logs: {server_log_dir}")

                print("--- Server is running and ready ---", flush=True)

                # 5. Loop through and Run Enabled Benchmarks
                stream_bench_logs = runner_config.get('stream_logs', False)

                for bench_run_config in benchmark_control.get('run', []):
                    if not bench_run_config.get('enabled', False):
                        print(f"\n--- Skipping Benchmark: {bench_run_config['type']} (disabled in config) ---", flush=True)
                        continue

                    benchmark_type = bench_run_config['type']
                    print(f"\n--- Running Benchmark: {benchmark_type} for {engine_tp_pp_id} ---", flush=True)

                    try:
                        # Get the benchmark command, log paths, and specific conda env
                        bench_cmd, bench_log_dir, bench_log_prefix, bench_conda_env = get_benchmark_command(
                            benchmark_config=bench_run_config,
                            engine_identifier=engine_tp_pp_id,
                            host=server_host, port=current_port,
                            run_base_dir=run_base_dir, # Pass the engine/tp/pp base dir
                            config=config,
                            main_config_path=main_config_path
                        )

                        # Set up logging for *this specific benchmark instance*
                        bench_stdout_log, bench_stderr_log = setup_logging(bench_log_dir, bench_log_prefix)

                        # Determine final conda env (benchmark specific, else server, else None)
                        final_conda_env = bench_conda_env if bench_conda_env is not None else server_conda_env_name

                        # Execute the benchmark command
                        run_command(
                            bench_cmd,
                            env_name=final_conda_env,
                            conda_base_env=runner_config.get('conda_base_env'),
                            popen=False, check=False, # Don't stop script on benchmark error
                            stdout_log_path=bench_stdout_log,
                            stderr_log_path=bench_stderr_log,
                            stream_logs=stream_bench_logs
                        )

                    except Exception as bench_e:
                        print(f"!!! Benchmark '{benchmark_type}' failed for {engine_tp_pp_id}: {bench_e} !!!", flush=True)
                        # Continue to next benchmark type even if one fails

            except Exception as e:
                print(f"\n!!! === ERROR during run for {engine_tp_pp_id}: {e} === !!!\n", flush=True)
                import traceback
                traceback.print_exc()

            finally:
                # 6. Kill Server Process Group
                print(f"\n--- Cleaning up server for {engine_tp_pp_id} ---", flush=True)
                kill_process_group_and_close_logs(server_process, server_stdout_f, server_stderr_f)
                server_process, server_stdout_f, server_stderr_f = None, None, None
                print(f"\n{'='*25} Finished Run: {engine_tp_pp_id} {'='*25}\n", flush=True)
                time.sleep(5)

    print("\nAll configured benchmark runs completed.", flush=True)


if __name__ == "__main__":
    main()
