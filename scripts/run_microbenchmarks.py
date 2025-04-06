import subprocess
import time
import os
import signal
import shlex
import argparse
import itertools
import requests # Added for server readiness check
from requests.exceptions import ConnectionError, Timeout
from typing import List, Tuple, Dict, Optional, Any

# --- Default Configuration ---

# Model paths/IDs
DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_CHAT_TEMPLATE = None # Make chat template optional by default

# Server Config
DEFAULT_SERVER_PORT = 8000
DEFAULT_SERVER_READINESS_TIMEOUT = 300 # Max seconds to wait for server to become ready
DEFAULT_SERVER_CHUNK_SIZE = 512

# Benchmark Config
DEFAULT_BENCHMARK_TIMEOUT = 600
DEFAULT_BASE_OUTPUT_DIR = "engine_microbenchmark_logs"

# Prefill Profiler Specific Args
DEFAULT_PREFILL_LENGTHS = [512, 1024, 2048, 4086]

# Decode Profiler Specific Args
DEFAULT_DECODE_CONTEXT_LENGTHS = [512]
DEFAULT_DECODE_BATCH_SIZES = [1, 8, 16, 32, 64, 128]

# Vajra Specific Defaults
DEFAULT_VAJRA_DTYPE = "float16"
DEFAULT_VAJRA_PRIORITIZER = "FCFS"
DEFAULT_VAJRA_SCHEDULER = "FIXED_CHUNK" # As per user fix

# --- Helper Functions ---

def setup_logging(log_dir: str, name_prefix: str) -> Tuple[str, str]:
    """Creates log directory and returns paths for stdout and stderr log files."""
    os.makedirs(log_dir, exist_ok=True)
    stdout_log_path = os.path.join(log_dir, f"{name_prefix}_stdout.log")
    stderr_log_path = os.path.join(log_dir, f"{name_prefix}_stderr.log")
    return stdout_log_path, stderr_log_path

def run_command(
    cmd_list: List[str],
    env_name: Optional[str] = None,
    popen: bool = False,
    check: bool = True,
    stdout_log_path: Optional[str] = None,
    stderr_log_path: Optional[str] = None,
    stream_logs: bool = False,
) -> Optional[Tuple[subprocess.Popen, Optional[Any], Optional[Any]]]:
    """
    Runs a command, optionally within a conda environment, prints it,
    handles execution, and manages logging.

    Returns:
        If popen=True: A tuple (Popen object, stdout file handle, stderr file handle)
                      File handles can be None if logging paths aren't provided.
        If popen=False: None
    """
    if env_name:
        full_cmd = ["conda", "run", "--no-capture-output", "-n", env_name] + cmd_list
        # Note: --no-capture-output with conda run might be needed depending on version
        # to ensure stdout/stderr are correctly piped when redirecting in Popen
    else:
        full_cmd = cmd_list

    cmd_str = ' '.join(shlex.quote(str(part)) for part in full_cmd)
    print(f"\nExecuting in env '{env_name or 'base'}': {cmd_str}", flush=True)
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
            # Start in background, redirect stdout/stderr to log files
            process = subprocess.Popen(
                full_cmd,
                preexec_fn=os.setsid,
                stdout=stdout_f,
                stderr=stderr_f
            )
            # Return process and file handles so they can be closed later
            return process, stdout_f, stderr_f
        else:
            # Run synchronously
            result = subprocess.run(
                full_cmd,
                check=False, # Check manually after logging
                text=True,
                capture_output=True # Capture for logging/streaming
            )

            # Write captured output to logs
            if stdout_f:
                stdout_f.write(result.stdout)
                stdout_f.flush() # Ensure it's written
            if stderr_f:
                stderr_f.write(result.stderr)
                stderr_f.flush() # Ensure it's written

            print(f"Command finished with exit code {result.returncode}.")

            # Optionally stream logs to console
            if stream_logs:
                print("\n--- Benchmark STDOUT ---")
                print(result.stdout)
                print("--- Benchmark STDERR ---")
                print(result.stderr)
                print("--- End Logs ---\n")
            else:
                 # Limit output length to avoid flooding console if not streaming fully
                stdout_suffix = "..." if len(result.stdout) > 500 else ""
                stderr_suffix = "..." if len(result.stderr) > 500 else ""
                print(f"STDOUT (last 500 chars):\n{result.stdout[-500:]}{stdout_suffix}")
                print(f"STDERR (last 500 chars):\n{result.stderr[-500:]}{stderr_suffix}")

            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, full_cmd, output=result.stdout, stderr=result.stderr)

            return None # Indicate synchronous completion

    except subprocess.CalledProcessError as e:
        print(f"!!! Command failed with exit code {e.returncode} !!! Logs are in the files above.", flush=True)
        if check: raise
        return None
    except FileNotFoundError as e:
        print(f"!!! Command or Conda environment not found: {e}. Is conda installed and in PATH? Is env '{env_name}' correct? !!!", flush=True)
        if popen: return None
        raise
    except Exception as e:
        print(f"!!! An unexpected error occurred while running command: {e} !!!", flush=True)
        if popen: return None # If Popen failed immediately
        raise
    finally:
        # Close log files ONLY if running synchronously (popen=False)
        # For popen=True, the file handles are returned and must be closed later
        if not popen:
            if stdout_f: stdout_f.close()
            if stderr_f: stderr_f.close()


def kill_process_group_and_close_logs(
    process: Optional[subprocess.Popen],
    stdout_log_f: Optional[Any],
    stderr_log_f: Optional[Any]
):
    """Reliably kills the process group and closes associated log file handles."""
    if process and process.poll() is None:
        pgid = 0
        try:
            pgid = os.getpgid(process.pid)
            print(f"Attempting to kill process group {pgid} (PID: {process.pid}) (SIGTERM)...", flush=True)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(5)

            if process.poll() is None:
                print(f"Process group {pgid} did not exit, sending SIGKILL...", flush=True)
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(2)

            process.wait(timeout=10)
            print(f"Process group {pgid} terminated.", flush=True)

        except ProcessLookupError:
            print(f"Process group {pgid} (PID: {process.pid}) already gone.", flush=True)
        except Exception as e:
            print(f"Error killing process group {pgid} (PID: {process.pid}): {e}", flush=True)
            # Fallback kill attempt
            try:
                if process.poll() is None: process.terminate()
                time.sleep(2)
                if process.poll() is None: process.kill()
                process.wait(timeout=5)
                print(f"Main process PID {process.pid} terminated (fallback).")
            except Exception as fallback_e:
                 print(f"Error during fallback kill of PID {process.pid}: {fallback_e}", flush=True)

    elif process:
         print(f"Server process (PID: {process.pid}) already terminated before kill attempt.", flush=True)
    else:
        print("No server process to kill.", flush=True)

    # Close log files
    if stdout_log_f:
        try: stdout_log_f.close()
        except Exception as e: print(f"Error closing stdout log: {e}", flush=True)
    if stderr_log_f:
        try: stderr_log_f.close()
        except Exception as e: print(f"Error closing stderr log: {e}", flush=True)

def wait_for_server_ready(host: str, port: int, timeout: int, check_endpoint: str = "/v1/models") -> bool:
    """Polls the server API endpoint until it's ready or timeout occurs."""
    start_time = time.monotonic()
    url = f"http://{host}:{port}{check_endpoint}"
    print(f"Waiting for server at {url} to be ready (timeout: {timeout}s)...", flush=True)
    while time.monotonic() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2) # Short timeout for individual check
            if response.status_code == 200:
                print(f"Server is ready! (responded in {time.monotonic() - start_time:.2f}s)", flush=True)
                return True
            else:
                print(f". (status: {response.status_code})", end='', flush=True)
        except ConnectionError:
            print(".", end='', flush=True) # Server not listening yet
        except Timeout:
            print("T", end='', flush=True) # Server listening but not responding quickly
        except Exception as e:
            print(f"E({e})", end='', flush=True) # Other errors

        time.sleep(2) # Wait before next poll

    print(f"\nServer readiness check failed after {timeout} seconds.", flush=True)
    return False


def get_server_command(
    engine: str,
    model_id: str,
    tp: int,
    pp: int,
    port: int,
    chunk_size: int,
    chat_template: Optional[str] = None,
    **kwargs: Any
) -> List[str]:
    """Builds the server start command for the specified engine."""
    # Consolidate args for easier access
    args = {
        "model": model_id, "tp": tp, "pp": pp, "port": port,
        "chunk_size": chunk_size, "chat_template": chat_template, **kwargs
    }

    if engine == "sglang":
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", args["model"],
            "--port", str(args["port"]),
            "--dtype", "auto",
            "--tensor-parallel-size", str(args["tp"]),
            "--chunked-prefill-size", str(args["chunk_size"]),
            "--log-level", "error",
        ]
    elif engine == "vllm":
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", args["model"],
            "--dtype", "auto",
            "--tensor-parallel-size", str(args["tp"]),
            "--pipeline-parallel-size", str(args["pp"]),
            "--max-num-batched-tokens", str(args["chunk_size"]),
            "--port", str(args["port"]),
            "--enable-chunked-prefill", "true",
            "--max-num-seqs", "256",
            "--uvicorn-log-level", "error",
            "--disable-log-stats",
            "--disable-log-requests",
        ]
        if args.get("chat_template"):
             cmd.extend(["--chat-template", args["chat_template"]])
    elif engine == "vajra":
         cmd = [
            "python", "-m", "vajra.entrypoints.openai.api_server",
            "--model_config_model", args["model"],
            "--model_config_dtype", args.get("vajra_dtype", DEFAULT_VAJRA_DTYPE),
            "--request_prioritizer_config_type", args.get("vajra_prioritizer", DEFAULT_VAJRA_PRIORITIZER).upper(),
            "--scheduler_config_type", args.get("vajra_scheduler", DEFAULT_VAJRA_SCHEDULER),
            "--parallel_config_tensor_parallel_size", str(args["tp"]),
            "--parallel_config_pipeline_parallel_size", str(args["pp"]),
            # Pass chunk sizes relevant to different schedulers
            "--fixed_chunk_replica_scheduler_config_chunk_size", str(args["chunk_size"]),
            "--port", str(args["port"]),
            "--host", "127.0.0.1",
            "--log_level", "error",
        ]
         if args.get("chat_template"):
             cmd.extend(["--chat_template", args["chat_template"]])
    else:
        raise ValueError(f"Unsupported engine: {engine}")

    return cmd

def get_benchmark_command(
    profile_type: str, # "prefill" or "decode"
    engine_identifier: str, # e.g., "sglang_tp1_pp1"
    model_id: str,
    output_dir_base: str,
    timeout: int,
    prefill_lengths: Optional[List[int]] = None,
    decode_context_lengths: Optional[List[int]] = None,
    decode_batch_sizes: Optional[List[int]] = None,
) -> Tuple[List[str], str, str]: # Returns command, output_dir, log_prefix
    """Builds the Veeksha profiler command, output directory, and log prefix."""

    run_output_dir = os.path.join(output_dir_base, f"{profile_type}_{engine_identifier}")
    log_prefix = f"{profile_type}_benchmark" # Filename prefix for benchmark logs

    common_args = [
        f"--client_config_model={model_id}",
        # --- IMPORTANT: Point client to the running server ---
        # ----------------------------------------------------
        f"--timeout={timeout}",
        f"--metrics_config_output_dir={run_output_dir}", # Veeksha's metrics output dir
        "--metrics_config_should_use_given_dir",
    ]

    if profile_type == "prefill":
        if not prefill_lengths: raise ValueError("Prefill lengths needed.")
        cmd = [
            "python", "-m", "veeksha.prefill_profiler", *common_args,
            "--no-prefill_profiler_config_should_train_predictor",
            "--prefill_profiler_config_prefill_lengths", *map(str, prefill_lengths),
        ]
    elif profile_type == "decode":
        if not decode_context_lengths or not decode_batch_sizes: raise ValueError("Decode lengths/batches needed.")
        cmd = [
            "python", "-m", "veeksha.decode_profiler", *common_args,
            "--decode_profiler_config_context_lengths", *map(str, decode_context_lengths),
            "--decode_profiler_config_batch_sizes", *map(str, decode_batch_sizes),
        ]
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return cmd, run_output_dir, log_prefix


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Run LLM Engine Benchmarks (SGLang, vLLM, Vajra) with Logging and Readiness Checks")

    # Engine and Parallelism
    parser.add_argument("--engines", nargs='+', required=True, choices=['sglang', 'vllm', 'vajra'], help="List of engines.")
    parser.add_argument("--tp-dims", nargs='+', type=int, required=True, help="List of Tensor Parallel dimensions.")
    parser.add_argument("--pp-dims", nargs='+', type=int, required=True, help="List of Pipeline Parallel dimensions.")
    parser.add_argument("--max-gpus", type=int, default=8, help="Skip runs where TP*PP > max_gpus.")

    # Model and Paths
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="Model identifier.")
    parser.add_argument("--chat-template", type=str, default=DEFAULT_CHAT_TEMPLATE, help="Optional: Path to chat template.")
    parser.add_argument("--base-output-dir", type=str, default=DEFAULT_BASE_OUTPUT_DIR, help="Base directory for results.")

    # Server Config
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the server.")
    parser.add_argument("--start-port", type=int, default=DEFAULT_SERVER_PORT, help="Starting server port.")
    parser.add_argument("--server-readiness-timeout", type=int, default=DEFAULT_SERVER_READINESS_TIMEOUT, help="Max seconds to wait for server API.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_SERVER_CHUNK_SIZE, help="Server chunk size.")

    # Benchmark Config
    parser.add_argument("--benchmark-timeout", type=int, default=DEFAULT_BENCHMARK_TIMEOUT, help="Timeout per benchmark.")
    parser.add_argument("--prefill-lengths", nargs='+', type=int, default=DEFAULT_PREFILL_LENGTHS, help="Prefill lengths.")
    parser.add_argument("--decode-context-lengths", nargs='+', type=int, default=DEFAULT_DECODE_CONTEXT_LENGTHS, help="Decode context lengths.")
    parser.add_argument("--decode-batch-sizes", nargs='+', type=int, default=DEFAULT_DECODE_BATCH_SIZES, help="Decode batch sizes.")

    # Environment and Logging Config
    parser.add_argument("--conda-base-env", type=str, default=None, help="Optional: Path to base conda env.")
    parser.add_argument("--env-prefix", type=str, default="", help="Optional: Prefix for conda env names.")
    parser.add_argument("--stream-logs", action='store_true', help="Stream benchmark stdout/stderr to console after completion.")

    args = parser.parse_args()

    os.makedirs(args.base_output_dir, exist_ok=True)
    current_port = args.start_port
    parallel_combinations = list(itertools.product(args.tp_dims, args.pp_dims))

    for engine in args.engines:
        for tp, pp in parallel_combinations:
            # Check GPU requirement
            if tp * pp > args.max_gpus:
                print(f"\n--- Skipping {engine} TP={tp}, PP={pp} (requires {tp*pp} GPUs > max {args.max_gpus}) ---\n", flush=True)
                continue

            if pp > 1 and engine == "sglang":
                print(f"\n--- Skipping {engine} TP={tp}, PP={pp} (PP > 1 is not supported for sglang) ---\n", flush=True)
                continue

            engine_tp_pp_id = f"{engine}_tp{tp}_pp{pp}"
            conda_env_name = f"{args.env_prefix}{engine}"
            run_base_dir = os.path.join(args.base_output_dir, engine_tp_pp_id) # Base dir for this specific run

            print(f"\n{'='*25} Starting Benchmark: {engine_tp_pp_id} {'='*25}\n", flush=True)
            server_process = None
            server_stdout_f, server_stderr_f = None, None # Hold server log file handles

            try:
                # 1. Setup Server Logging
                server_log_dir = os.path.join(run_base_dir, "server_logs")
                server_stdout_log, server_stderr_log = setup_logging(server_log_dir, "server")

                # 2. Construct and Start Server Command
                print(f"--- Launching {engine} Server (TP={tp}, PP={pp}) ---", flush=True)
                server_cmd_list = get_server_command(
                    engine=engine, model_id=args.model_id, tp=tp, pp=pp,
                    port=current_port, chunk_size=args.chunk_size,
                    chat_template=args.chat_template,
                    # Pass other engine-specific defaults/args if needed
                    vajra_dtype=DEFAULT_VAJRA_DTYPE,
                    vajra_prioritizer=DEFAULT_VAJRA_PRIORITIZER,
                    vajra_scheduler=DEFAULT_VAJRA_SCHEDULER,
                )
                server_run_result = run_command(
                    server_cmd_list, env_name=conda_env_name,
                    popen=True, stdout_log_path=server_stdout_log,
                    stderr_log_path=server_stderr_log
                )

                if server_run_result is None: # Popen failed immediately
                    raise RuntimeError(f"Failed to start server process for {engine_tp_pp_id}. Check logs.")
                server_process, server_stdout_f, server_stderr_f = server_run_result # Unpack

                # 3. Wait for Server Readiness (Dynamic Check)
                if not wait_for_server_ready(args.host, current_port, args.server_readiness_timeout):
                     # Check server logs immediately if it failed
                     if server_process.poll() is not None:
                         exit_code = server_process.returncode
                         raise RuntimeError(f"{engine} server (PID: {server_process.pid}) exited prematurely with code {exit_code} during readiness check for {engine_tp_pp_id}. Check server logs: {server_log_dir}")
                     else:
                        raise RuntimeError(f"{engine} server (PID: {server_process.pid}) did not become ready within {args.server_readiness_timeout}s for {engine_tp_pp_id}. Check server logs: {server_log_dir}")

                # Check if server died *after* readiness check started but before benchmarks
                if server_process.poll() is not None:
                    raise RuntimeError(f"{engine} server (PID: {server_process.pid}) died unexpectedly after becoming ready. Exit code: {server_process.returncode}. Check server logs: {server_log_dir}")

                print("Server is running and ready for benchmarks.", flush=True)


                # 4. Construct and Run Prefill Benchmark
                print(f"\n--- Running Prefill Benchmark for {engine_tp_pp_id} ---", flush=True)
                prefill_cmd, prefill_out_dir, prefill_log_prefix = get_benchmark_command(
                    profile_type="prefill", engine_identifier=engine_tp_pp_id,
                    model_id=args.model_id,
                    output_dir_base=args.base_output_dir, timeout=args.benchmark_timeout,
                    prefill_lengths=args.prefill_lengths,
                )
                prefill_stdout_log, prefill_stderr_log = setup_logging(prefill_out_dir, prefill_log_prefix)
                run_command(
                    prefill_cmd, env_name=conda_env_name, popen=False, check=False, # Don't halt script on benchmark failure
                    stdout_log_path=prefill_stdout_log, stderr_log_path=prefill_stderr_log,
                    stream_logs=args.stream_logs
                )

                # 5. Construct and Run Decode Benchmark
                print(f"\n--- Running Decode Benchmark for {engine_tp_pp_id} ---", flush=True)
                decode_cmd, decode_out_dir, decode_log_prefix = get_benchmark_command(
                    profile_type="decode", engine_identifier=engine_tp_pp_id,
                    model_id=args.model_id,
                    output_dir_base=args.base_output_dir, timeout=args.benchmark_timeout,
                    decode_context_lengths=args.decode_context_lengths,
                    decode_batch_sizes=args.decode_batch_sizes,
                )
                decode_stdout_log, decode_stderr_log = setup_logging(decode_out_dir, decode_log_prefix)
                run_command(
                    decode_cmd, env_name=conda_env_name, popen=False, check=False, # Don't halt script on benchmark failure
                    stdout_log_path=decode_stdout_log, stderr_log_path=decode_stderr_log,
                    stream_logs=args.stream_logs
                )

            except Exception as e:
                print(f"\n!!! === ERROR during benchmark for {engine_tp_pp_id}: {e} === !!!\n", flush=True)
                # Optional: Add traceback print here:
                # import traceback
                # traceback.print_exc()

            finally:
                # 6. Kill Server Process Group and Close Log Files
                print(f"\n--- Cleaning up for {engine_tp_pp_id} ---", flush=True)
                kill_process_group_and_close_logs(server_process, server_stdout_f, server_stderr_f)
                server_process, server_stdout_f, server_stderr_f = None, None, None # Clear state
                # Optional: Increment port for next server instance
                # current_port += 1
                print(f"\n{'='*25} Finished Benchmark: {engine_tp_pp_id} {'='*25}\n", flush=True)
                time.sleep(3) # Shorter pause needed now

    print("\nAll benchmark iterations completed.", flush=True)

if __name__ == "__main__":
    main()
