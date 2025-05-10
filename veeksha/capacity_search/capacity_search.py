import argparse
import glob
import json
import subprocess
import os
import threading
import signal
import requests
import traceback
from typing import Optional, Union, List, Tuple, Dict, Any, IO, TextIO
from requests.exceptions import ConnectionError
import time
import yaml
import re
import shutil

import numpy as np
import wandb

from veeksha.capacity_search.benchmark_wrapper import run
from veeksha.capacity_search.config.config import BenchmarkConfig, JobConfig, _get_hash
from veeksha.logger import init_logger

logger = init_logger(__name__)

# Increase upper bound of QPS by this scale during binary search
QPS_INCREASE_SCALE = 2
# Threshold to increase the upper bound of QPS during binary search
VICINITY_THRESHOLD = 0.8


class LogMonitorThread(threading.Thread):
    """Thread class to continuously monitor server logs for errors."""
    
    def __init__(self, 
                 stdout_file_path: str, 
                 stderr_file_path: str, 
                 error_patterns: List[str],
                 check_interval: int = 5,
                 callback=None,
                 skip_iteration_flag=None):
        """Initialize the log monitor thread.
        
        Args:
            stdout_file_path: Path to the stdout log file
            stderr_file_path: Path to the stderr log file
            error_patterns: List of regex patterns to check for errors
            check_interval: How often to check logs (in seconds)
            callback: Optional callback function when error is found, receives error_msg as parameter
            skip_iteration_flag: An Event object that can be set to signal the main loop to skip an iteration
        """
        super().__init__(daemon=True)  # daemon=True means thread will exit when main thread exits
        self.stdout_file_path = stdout_file_path
        self.stderr_file_path = stderr_file_path
        self.error_patterns = error_patterns
        self.check_interval = check_interval
        self.callback = callback
        self.skip_iteration_flag = skip_iteration_flag
        self.running = False
        self.last_stdout_pos = 0
        self.last_stderr_pos = 0
        self._stop_event = threading.Event()
    
    def stop(self):
        """Stop the monitoring thread."""
        self._stop_event.set()
        self.running = False
    
    def run(self):
        """Main thread loop that periodically checks logs for errors."""
        self.running = True
        stdout_file = None
        stderr_file = None
        
        try:
            stdout_file = open(self.stdout_file_path, "r")
            stderr_file = open(self.stderr_file_path, "r")
            
            while not self._stop_event.is_set() and self.running:
                # Check for errors in logs
                found_error, error_msg, self.last_stdout_pos, self.last_stderr_pos = check_server_logs_for_errors(
                    stdout_file, stderr_file, self.error_patterns, 
                    self.last_stdout_pos, self.last_stderr_pos
                )                
                # If error found, set the skip_iteration_flag and call the callback if provided
                if found_error:
                    if self.skip_iteration_flag:
                        print(f"LogMonitorThread detected error, signaling to skip iteration: {error_msg}", flush=True)
                        self.skip_iteration_flag.set()
                        
                    if self.callback:
                        self.callback(error_msg)
                    else:
                        print(f"LogMonitorThread detected error: {error_msg}", flush=True)
                
                # Wait for next check interval
                self._stop_event.wait(self.check_interval)
                
        except Exception as e:
            print(f"Error in log monitoring thread: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Close files when thread exits
            if stdout_file and not stdout_file.closed:
                stdout_file.close()
            if stderr_file and not stderr_file.closed:
                stderr_file.close()


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


class CapacitySearch:
    def __init__(
        self,
        job_config: JobConfig,
        args: argparse.Namespace,
    ) -> None:
        self.job_config = job_config
        self.args = args

        if (self.args.slo_type == "deadline") and self.args.dynamic_ttft_slo:
            assert (
                self.args.profile_dir is not None
            ), "Deadline SLO needs profiled predictions"

    def _run_benchmark(self, benchmark_config: BenchmarkConfig):
        run(self.job_config, benchmark_config)

    def _get_result_file(self, run_dir: str, metric_name: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"{metric_name}.csv"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_request_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"request_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _get_service_level_metrics(self, run_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(run_dir, f"service_level_metrics.json"))
        if len(files) == 0:
            return None

        return files[0]

    def _use_deadline_based_slo(
        self, request_level_metrics_file: str
    ) -> Tuple[bool, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        deadline_miss_rate_array = request_level_metrics["deadline_miss_rate"]

        # Calculate percentile values of deadline miss rate
        deadline_miss_rate = np.quantile(
            deadline_miss_rate_array, self.args.deadline_miss_rate_percentile
        )

        is_under_sla = deadline_miss_rate <= self.args.deadline_miss_rate_slo

        return is_under_sla, deadline_miss_rate

    def _use_tbt_and_ttft_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TBT request level
        ttft_array = request_level_metrics["ttft"]
        tbt_array = request_level_metrics["tbt"]

        # Merge TBT arrays of each request to make it service level
        combined_tbt_array = []
        for i in range(len(tbt_array)):
            combined_tbt_array.extend(tbt_array[i])

        # Calculate percentile values of TBT, TTFT
        tbt = np.quantile(combined_tbt_array, self.args.tbt_percentile)
        ttft = np.quantile(ttft_array, self.args.ttft_percentile)

        is_under_sla = tbt <= self.args.tbt_slo and ttft <= self.args.ttft_slo

        return is_under_sla, tbt, ttft

    def _use_ttft_and_tpot_slo(
        self,
        request_level_metrics_file: str,
    ) -> Tuple[bool, float, float]:
        with open(request_level_metrics_file, "r") as f:
            request_level_metrics = json.load(f)

        # Get TTFT, TPOT at request level
        ttft_array = request_level_metrics["ttft"]
        tpot_array = request_level_metrics["tpot"]

        # Calculate percentile values of TTFT, TPOT
        ttft = np.quantile(ttft_array, self.args.ttft_percentile)
        tpot = np.quantile(tpot_array, self.args.tpot_percentile)

        is_under_sla = ttft <= self.args.ttft_slo and tpot <= self.args.tpot_slo

        return is_under_sla, ttft, tpot

    def _is_under_sla(
        self,
        request_level_metrics_file: str,
        benchmark_config: BenchmarkConfig,
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
        is_under_sla = False
        tbt = None
        ttft = None
        tpot = None
        deadline_miss_rate = None

        if self.args.slo_type == "deadline":
            is_under_sla, deadline_miss_rate = self._use_deadline_based_slo(
                request_level_metrics_file
            )
        elif self.args.slo_type == "tbt_ttft":
            is_under_sla, tbt, ttft = self._use_tbt_and_ttft_slo(
                request_level_metrics_file
            )
        elif self.args.slo_type == "ttft_tpot":
            is_under_sla, ttft, tpot = self._use_ttft_and_tpot_slo(
                request_level_metrics_file
            )
        else:
            raise ValueError(f"Invalid SLO type: {self.args.slo_type}")

        print(
            f"{benchmark_config.to_human_readable_name()}"
            f" - TBT P{self.args.tbt_percentile * 100} Tokens: {tbt}"
            f" - TTFT P{self.args.ttft_percentile * 100} Tokens: {ttft}"
            f" - TPOT P{self.args.tpot_percentile * 100} Requests: {tpot}"
            f" - Deadline Miss Rate P{self.args.deadline_miss_rate_percentile * 100} Requests: {deadline_miss_rate}",
        )
        return (
            is_under_sla,
            tbt,
            ttft,
            tpot,
            deadline_miss_rate,
            benchmark_config.get_run_id(),
        )

    def is_under_sla(
        self, qps: float, trace_input_file: str
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:

        # replace "processed" with "generated"
        trace_input_file = trace_input_file.replace("processed", "generated")

        self.job_config.request_generator_config.trace_request_length_generator_trace_file = trace_input_file
        self.job_config.request_generator_config.trace_request_interval_generator_trace_file = trace_input_file

        job_config_key = self.job_config.get_key()
        slo_key = "tbtslo{}_ttftslo{}_tpotslo{}_ttftslackslo{}_deadlinemissrateslo{}_dynamicttftslo{}".format(
            self.args.tbt_slo,
            self.args.ttft_slo,
            self.args.tpot_slo,
            self.args.ttft_slack_slo,
            self.args.deadline_miss_rate_slo,
            self.args.dynamic_ttft_slo,
        )
        overall_key = "_".join([job_config_key, slo_key])
        # since key is very long, hash it to get a unique key for a particular config
        # just check config.json to know actual config
        hash_key = _get_hash(overall_key)

        benchmark_config = BenchmarkConfig(
            output_dir=os.path.join(
                self.args.output_dir,
                str(self.job_config.server_config.openai_server_engine),
                self.job_config.model_config.name,
                # f"ttft_slack_{self.args.ttft_slack_slo}_tbt_{self.args.tbt_slo}",
                trace_input_file,# str(self.job_config.request_generator_config.trace_file_name),
                f"{hash_key}_q{qps}",
            ),
            qps=qps,
            tbt_deadline=self.args.tbt_slo,
            ttft_deadline=self.args.ttft_slo,
            ttft_slack=self.args.ttft_slack_slo,
            wandb_project=self.args.wandb_project,
            wandb_group=self.args.wandb_group,
            wandb_run_name=f"qps_{qps}_model_{self.job_config.model_config.name}_engine_{self.job_config.server_config.openai_server_engine}",
            should_write_metrics=self.args.should_write_metrics_to_wandb,
            use_predictions_for_ttft=(self.args.slo_type == "deadline")
            and self.args.dynamic_ttft_slo,
            predictor_dir=self.args.profile_dir,
        )

        run_dir = benchmark_config.get_run_dir()
        os.makedirs(run_dir, exist_ok=True)

        cached_request_level_metrics_file = self._get_request_level_metrics(run_dir)

        if cached_request_level_metrics_file is not None:
            print(f"Cached results found for {qps}")
            return self._is_under_sla(
                cached_request_level_metrics_file, benchmark_config
            )

        self._run_benchmark(benchmark_config)

        request_level_metrics_file = self._get_request_level_metrics(run_dir)

        assert (
            request_level_metrics_file is not None
        ), f"Service-level metrics file not found for {benchmark_config.to_human_readable_name()}"

        return self._is_under_sla(request_level_metrics_file, benchmark_config)

    def is_server_up(self, host="localhost", port=8000, max_retries=10, retry_interval=3):
        """Check if the SGLang server is up and responding."""
        url = f"http://{host}:{port}/v1/completions"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Server is up after {attempt+1} attempts!")
                    return True
                else:
                    print(f"Server responded with status code {response.status_code}, retrying...")
            except (requests.ConnectionError, requests.Timeout) as e:
                print(f"Attempt {attempt+1}/{max_retries}: Server not ready yet ({str(e)})")
            
            if attempt < max_retries - 1:  # Don't sleep after the last attempt
                time.sleep(retry_interval)
        
        print(f"Server failed to come up after {max_retries} attempts")
        return False

    def is_port_in_use(self, port, host='localhost'):
        """Check if a port is already in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    def _get_directory_state(self, dir_path: str) -> Tuple[int, int]:
        """Helper to get item count and total size of files in a directory."""
        item_count = 0
        total_size = 0
        if not os.path.isdir(dir_path):
            return 0, 0
        try:
            for root, _, files in os.walk(dir_path):
                item_count += len(files) # Consider if dirs should be counted too
                for f_name in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, f_name))
                    except OSError:
                        pass # File might be gone or inaccessible during walk
        except Exception as e:
            logger.warning(f"Error walking directory {dir_path} for state: {e}")
            return -1, -1 # Indicate error
        return item_count, total_size

    def search(self):
        """
        Perform binary search to find the maximum QPS under the SLO
        """

        print(
            f"Starting search for {self.job_config.get_human_readable_name()}",
        )

        left = 0
        right = self.job_config.start_qps * 2
        qps = 0
        last_qps = 0
        max_qps_under_sla = None
        min_qps_over_sla = 2**32

        tbt_at_max_qps = None
        ttft_at_max_qps = None
        tpot_at_max_qps = None
        deadline_miss_rate_at_max_qps = None
        best_run_id = None
        found_valid_qps = False

        first_server_launch = True
        with open(self.args.server_launch_file, 'r') as f:
            server_config = yaml.safe_load(f)

        try:
            enable_cache_telemetry = server_config["enable_cache_telemetry"]
        except KeyError:
            enable_cache_telemetry = False

        source_trace_file = self.job_config.request_generator_config.trace_request_length_generator_trace_file

        for iteration in range(self.args.max_iterations):
            print(f"=== Starting iteration {iteration + 1}/{self.args.max_iterations} ===")
            print(f"Search space: left={left:.2f} QPS, right={right:.2f} QPS")

            # stopping condition - we have reached the minimum granularity
            if abs(left - right) < self.args.min_search_granularity * qps / 100:
                print(f"Search granularity {abs(left - right):.2f} below minimum threshold")
                stdout_file.close()
                stderr_file.close()
                break

            qps = (left + right) / 2
            # round to 2 decimal places
            qps = round(qps, 2)

            cmd_generate_trace = f"python experiments/generate_session_sampled_trace.py --trace-file {source_trace_file} --minimum-match-threshold {self.args.session_match_threshold} --dispatch-rate {qps} --max-context-length {self.job_config.request_generator_config.request_generator_max_tokens}"
            subprocess.run(cmd_generate_trace, shell=True, check=True)

            trace_input_file = f"{source_trace_file}/{qps}/sampled_trace_dr{qps}_mmt{self.args.session_match_threshold}.jsonl"

            if qps == last_qps:
                print(f"QPS {qps} unchanged from previous iteration, stopping search")
                stdout_file.close()
                stderr_file.close()
                break

            last_qps = qps
            print(f"Testing QPS: {qps}")

            slo_key = "tbtslo{}_ttftslo{}_tpotslo{}_ttftslackslo{}_deadlinemissrateslo{}_dynamicttftslo{}".format(
                        self.args.tbt_slo,
                        self.args.ttft_slo,
                        self.args.tpot_slo,
                        self.args.ttft_slack_slo,
                        self.args.deadline_miss_rate_slo,
                        self.args.dynamic_ttft_slo,
            )
            overall_key = "_".join([self.job_config.get_key(), slo_key])
            # since key is very long, hash it to get a unique key for a particular config
            # just check config.json to know actual config
            hash_key = _get_hash(overall_key)
            run_dir = os.path.join(
                self.args.output_dir,
                str(self.job_config.server_config.openai_server_engine),
                self.job_config.model_config.name,
                # f"ttft_slack_{self.args.ttft_slack_slo}_tbt_{self.args.tbt_slo}",
                trace_input_file,# str(self.job_config.request_generator_config.trace_file_name),
                f"{hash_key}_q{qps}",
            )

            print(f"Run directory: {run_dir}")

            if self.job_config.request_generator_config.request_interval_generator_provider == "trace" and self.job_config.request_generator_config.request_length_generator_provider == "trace":
                run_dir = run_dir.replace("processed_traces", "generated_traces")

            cached_request_level_metrics_file = self._get_request_level_metrics(run_dir)
            print(f"Cached request level metrics file: {cached_request_level_metrics_file}")
            if cached_request_level_metrics_file is None:
                print(f"Cache for qps {qps} not found, starting server")

                ############## server launching. we restart the server on each iteration
                if True: #enable_cache_telemetry or first_server_launch:
                    # first_server_launch = False
                    # Define the command and arguments as a list
                    if self.args.server_launch_file is not None:
                        print(f"Loading server config from {self.args.server_launch_file}")
                        # Load server configuration from YAML file
                        with open(self.args.server_launch_file, 'r') as f:
                            server_config = yaml.safe_load(f)
                        
                        print("Constructing server command from config")
                        # Construct command from YAML configuration

                        if "vllm" in server_config["module"]:
                            cmd = ["vllm", "serve", server_config["model"]]
                            # remove model from config
                            del server_config["model"]
                        else:
                            cmd = ["python", "-m", server_config["module"]]

                        port = 8000
                        while self.is_port_in_use(port):
                            port += 1

                        server_config["port"] = port

                        is_vajra = "vajra" in server_config["module"]
                        is_sglang = "sglang" in server_config["module"]
                        is_vllm = "vllm" in server_config["module"]
                        
                        # Add all configuration parameters from YAML
                        for key, value in server_config.items():
                            if key == "module" or key == "error_patterns" or key == "readiness_timeout": 
                                continue
                            elif key == "environment_variables":
                                # set environment variables with the values from the config {'LMCACHE_USE_EXPERIMENTAL': True, 'LMCACHE_CHUNK_SIZE': 256, 'LMCACHE_LOCAL_CPU': True, 'LMCACHE_MAX_LOCAL_CPU_SIZE': 80.0}
                                for k, v in value.items():
                                    os.environ[k] = str(v)
                            elif key in ["json_model_override_args", "hf_overrides"]:
                                if is_sglang:
                                    key = key.replace("_", "-")
                                cmd.extend(["--" + key, json.dumps(value)])
                            else:
                                # For vajra, keep underscores; for others, replace with hyphens
                                param_key = key if is_vajra else key.replace("_", "-")
                                
                                if isinstance(value, bool) and value:
                                    if is_vajra:
                                        cmd.extend([f"--{param_key}", str(value).lower()])
                                    else:
                                        cmd.append(f"--{param_key}")
                                elif not isinstance(value, bool):
                                    cmd.extend([f"--{param_key}", str(value)])
                    else:
                        raise ValueError("Server launch file not specified")


                    if self.job_config.server_config.server_env:
                        cmd = ["mamba", "run", "--no-capture-output", "-p", self.job_config.server_config.server_env] + cmd

                    print(f"Final command: {' '.join(cmd)}")
                    print(f"Starting server process on port {port}")
                    self.job_config.server_config.openai_api_url = f"http://localhost:{port}/v1"

                    try:
                        print(f"Opening log files in {self.args.output_dir}")
                        # Open log files for stdout and stderr
                        stdout_file = open(f"{self.args.output_dir}/server_stdout.log", "w")
                        stderr_file = open(f"{self.args.output_dir}/server_stderr.log", "w")
                        
                        # Redirect output to files
                        start_time = time.time()
                        print("Launching server subprocess")
                        
                        # Extract environment variables from server_config if they exist
                        env = os.environ.copy()  # Start with current environment
                        if 'environment_variables' in server_config:
                            print(f"Setting environment variables from server_config")
                            for key, value in server_config['environment_variables'].items():
                                env[key] = str(value)
                                print(f"  {key}={value}")
                        
                        server_process = subprocess.Popen(
                            cmd,
                            stdout=stdout_file,  # Redirect stdout to file
                            stderr=stderr_file,  # Redirect stderr to file
                            text=True,
                            env=env,  # Set environment variables
                            preexec_fn=os.setsid  
                        )
                        
                        # Check if process started successfully
                        if server_process.poll() is not None:
                            logger.error(f"Server process failed immediately with exit code: {server_process.returncode}")
                            continue
                            
                        pid = server_process.pid
                        print(f"Server process started successfully with PID: {pid}")

                    except Exception as e:
                        logger.error(f"Failed to start server process: {str(e)}", exc_info=True)
                        continue

                    error_patterns = server_config.get('error_patterns', [])
                    detected_fatal_error = False
                    # Start a log monitor thread to continuously check for errors in the background
                    if error_patterns:
                        def error_callback(error_msg):
                            print(f"\n!!! Log monitor detected fatal server error pattern: '{error_msg}' in logs for {self.job_config.server_config.openai_server_engine} !!!", flush=True)
                            detected_fatal_error = True
                        
                        # Create and assign log monitor
                        log_monitor = LogMonitorThread(
                            stdout_file_path=f"{self.args.output_dir}/server_stdout.log",
                            stderr_file_path=f"{self.args.output_dir}/server_stderr.log",
                            error_patterns=error_patterns,
                            check_interval=5,  # Check logs every 5 seconds
                            callback=error_callback
                        )
                        log_monitor.start()
                        print(f"Started background log monitoring thread for server logs", flush=True)
                        
                    # poll server until it's up or we timeout
                    try:
                        stdout_file_r = open(f"{self.args.output_dir}/server_stdout.log", "r")
                        stderr_file_r = open(f"{self.args.output_dir}/server_stderr.log", "r")
                        while time.monotonic() - start_time < server_config['readiness_timeout'] and not detected_fatal_error:
                            if server_process.poll() is not None: # Check 1: Process died?
                                raise RuntimeError(f"{self.job_config.server_config.openai_server_engine} server (PID: {pid}) exited prematurely (code {server_process.returncode}).")
                            
                            # Check 3: API Ready?
                            api_check_url = f"http://localhost:{server_config['port']}/v1/models"
                            try:
                                response = requests.get(api_check_url, timeout=2)
                                if response.status_code == 200:
                                    print(f"\nServer API is ready! ({api_check_url} responded {response.status_code} in {time.monotonic() - start_time:.2f}s)", flush=True)
                                    server_ready_status = True
                                    break
                                else: print(f"S({response.status_code})", end='', flush=True)
                            except ConnectionError: print(".", end='', flush=True)
                            except Exception as api_e: print(f"E({type(api_e).__name__})", end='', flush=True)

                            time.sleep(2)
                    finally:
                        if stdout_file_r: stdout_file_r.close()
                        if stderr_file_r: stderr_file_r.close()

            (
                is_under_sla,
                tbt,
                ttft,
                tpot,
                deadline_miss_rate,
                run_id,
            ) = self.is_under_sla(qps, trace_input_file)

            if is_under_sla:
                found_valid_qps = True
                max_qps_under_sla = qps
                tbt_at_max_qps = tbt
                ttft_at_max_qps = ttft
                tpot_at_max_qps = tpot
                deadline_miss_rate_at_max_qps = deadline_miss_rate
                best_run_id = run_id
                print(f"Found valid QPS={qps}, updating search bounds")

                if qps > VICINITY_THRESHOLD * right:
                    right = min(right * QPS_INCREASE_SCALE, min_qps_over_sla)
                    print(f"Expanding right bound to {right}")

                left = qps
            else:
                right = qps
                min_qps_over_sla = min(min_qps_over_sla, qps)
                print(f"QPS={qps} exceeded SLA, updating right bound to {right}")

            # get output_cache data and tag with current qps. Rename so that it's not overwritten by next run
            
            if enable_cache_telemetry:
                cache_output_path = self.args.cache_telemetry_path
                two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_cache_file = os.path.join(two_levels_up, cache_output_path)
                new_output_cache_file = None
                new_output_cache_file_ts = None

                if not os.path.exists(output_cache_file):
                    logger.warning(f"Output cache file {output_cache_file} does not exist")
                    continue

                with open(output_cache_file, "r") as f:
                    output_cache = json.load(f)

                # delete previous cache telemtry data to restart
                os.remove(output_cache_file)
                
                # tag with current qps and save
                output_cache["qps"] = qps
                cache_path_parts = os.path.splitext(cache_output_path)
                modified_cache_path = f"{cache_path_parts[0]}_qps_{qps}{cache_path_parts[1]}"
                new_output_cache_file = os.path.join(two_levels_up, modified_cache_path)
                
                with open(new_output_cache_file, "w") as f:
                    json.dump(output_cache, f)

                # when the telemetry is reset a cache telemetry file with timestamps is created (this is to avoid writing big files every 5 seconds).
                # we also get this file and tag it with qps
                output_cache_file_ts = output_cache_file.replace(".json", "_ts.json")

                # wait until the cache timeseries is written
                try:
                    start_time = time.monotonic()
                    # First wait for the file to exist
                    while not os.path.exists(output_cache_file_ts):
                        if time.monotonic() - start_time > 180:
                            raise RuntimeError("Cache timeseries file not written after 180 seconds")
                        time.sleep(1)
                    
                    # Then wait for the file size to stabilize, indicating complete write
                    last_size = -1
                    stable_count = 0
                    while stable_count < 10:  # Wait for size to be stable for 10 consecutive checks
                        current_size = os.path.getsize(output_cache_file_ts)
                        if current_size == last_size:
                            stable_count += 1
                        else:
                            stable_count = 0
                            last_size = current_size
                        
                        if time.monotonic() - start_time > 180:  # 3 minute timeout
                            raise RuntimeError("Cache timeseries file size not stabilized after 3 minutes")
                        time.sleep(1)
                    
                    # Now it's safe to read the file
                    with open(output_cache_file_ts, "r") as f:
                        output_cache = json.load(f)
                    output_cache["qps"] = qps
                    # tag filename
                    cache_path_parts = os.path.splitext(output_cache_file_ts)
                    modified_cache_path = f"{cache_path_parts[0]}_qps_{qps}{cache_path_parts[1]}"
                    new_output_cache_file_ts = os.path.join(two_levels_up, modified_cache_path)
                    with open(new_output_cache_file_ts, "w") as f:
                        json.dump(output_cache, f)
                    # remove it
                    os.remove(output_cache_file_ts)
                except Exception as e:
                    logger.error(f"Failed to wait for cache timeseries file: {str(e)}", exc_info=True)

            print(f"Cached request level metrics file: {cached_request_level_metrics_file}")
            if cached_request_level_metrics_file is None:
                try:
                    print(f"Terminating server process group {pid}")
                    # Kill the entire process group (server and any child processes it spawned)
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    
                    # Give it a moment to shut down gracefully
                    time.sleep(30)
                    
                    # If it's still running, force kill
                    if server_process.poll() is None:
                        logger.warning("Server didn't terminate gracefully, sending SIGKILL")
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        time.sleep(1)

                    if stdout_file: stdout_file.close()
                    if stderr_file: stderr_file.close()
                    
                    print(f"Server terminated with exit code: {server_process.returncode}")

                    if "vajra" in server_config["module"]:
                        logger.info("Running Vajra-specific telemetry tagging logic (CWD path strategy).")
                        relative_metrics_output_dir_from_yaml = server_config.get("metrics_config_output_dir")
                        logger.info(f"  Raw metrics_config_output_dir from server_config: {relative_metrics_output_dir_from_yaml}")

                        if relative_metrics_output_dir_from_yaml:
                            path_parts = relative_metrics_output_dir_from_yaml.split("veeksha/", 1)
                            if len(path_parts) > 1:
                                path_suffix_after_veeksha = path_parts[1]
                                logger.info(f"  Path segment extracted after 'veeksha/': {path_suffix_after_veeksha}")
                                
                                current_working_dir = os.getcwd()
                                logger.info(f"  Current Working Directory (CWD): {current_working_dir}")
                                
                                absolute_metrics_output_dir = os.path.join(current_working_dir, path_suffix_after_veeksha)
                                absolute_metrics_output_dir = os.path.normpath(absolute_metrics_output_dir)
                                logger.info(f"  Constructed absolute_metrics_output_dir: {absolute_metrics_output_dir}")

                                logger.info(f"  Checking existence of: {absolute_metrics_output_dir}")
                                if os.path.exists(absolute_metrics_output_dir) and os.path.isdir(absolute_metrics_output_dir):
                                    logger.info(f"  Metrics output directory {absolute_metrics_output_dir} exists. Waiting for file writing to stabilize before moving.")
                                    
                                    # Wait for directory stability
                                    STABILITY_CHECKS_REQUIRED = 5
                                    DIRECTORY_WAIT_TIMEOUT_SECONDS = 180 # 3 minutes
                                    wait_start_time = time.monotonic()
                                    last_item_count, last_total_size = -1, -1
                                    stable_checks_count = 0

                                    while stable_checks_count < STABILITY_CHECKS_REQUIRED:
                                        if time.monotonic() - wait_start_time > DIRECTORY_WAIT_TIMEOUT_SECONDS:
                                            logger.warning(f"Timeout after {DIRECTORY_WAIT_TIMEOUT_SECONDS}s waiting for directory {absolute_metrics_output_dir} to stabilize. Proceeding with current contents.")
                                            break
                                        
                                        current_item_count, current_total_size = self._get_directory_state(absolute_metrics_output_dir)
                                        logger.debug(f"    Stability check for {absolute_metrics_output_dir}: Items={current_item_count}, Size={current_total_size}. Stable checks: {stable_checks_count}/{STABILITY_CHECKS_REQUIRED}")

                                        if current_item_count == last_item_count and current_total_size == last_total_size and current_item_count != -1:
                                            stable_checks_count += 1
                                        else:
                                            stable_checks_count = 0
                                        
                                        last_item_count = current_item_count
                                        last_total_size = current_total_size
                                        
                                        if stable_checks_count < STABILITY_CHECKS_REQUIRED:
                                            time.sleep(2)

                                    if stable_checks_count >= STABILITY_CHECKS_REQUIRED:
                                        logger.info(f"  Directory {absolute_metrics_output_dir} stabilized with {last_item_count} items and total size {last_total_size} bytes.")
                                    # Proceed regardless of timeout, with warning if it occurred.

                                    logger.info(f"  Proceeding to move contents from stabilized directory: {absolute_metrics_output_dir}")
                                    parent_dir_of_metrics_output = os.path.dirname(absolute_metrics_output_dir)
                                    new_qps_specific_dir_name = f"{qps}" # User confirmed this naming (no 'qps_')
                                    new_qps_specific_dir_path = os.path.join(parent_dir_of_metrics_output, new_qps_specific_dir_name)
                                    logger.info(f"  Parent directory for QPS-specific folder: {parent_dir_of_metrics_output}")
                                    logger.info(f"  New QPS-specific directory name: {new_qps_specific_dir_name}")
                                    logger.info(f"  New QPS-specific directory full path: {new_qps_specific_dir_path}")

                                    logger.info(f"  Creating directory (if not exists): {new_qps_specific_dir_path}")
                                    os.makedirs(new_qps_specific_dir_path, exist_ok=True)

                                    logger.info(f"  Listing items in {absolute_metrics_output_dir} to move.")
                                    items_in_source_before_move = []
                                    if os.path.exists(absolute_metrics_output_dir):
                                        items_in_source_before_move = os.listdir(absolute_metrics_output_dir)
                                    logger.info(f"    Items found in source dir '{absolute_metrics_output_dir}': {items_in_source_before_move}")

                                    for item_name in items_in_source_before_move: # Iterate over the captured list
                                        source_item_path = os.path.join(absolute_metrics_output_dir, item_name)
                                        destination_item_path = os.path.join(new_qps_specific_dir_path, item_name)
                                        
                                        logger.info(f"    Processing item: '{item_name}'")
                                        logger.info(f"      Source path: {source_item_path}")
                                        logger.info(f"      Destination path: {destination_item_path}")
                                        logger.info(f"      Source exists before move? {os.path.exists(source_item_path)}")
                                        logger.info(f"      Source is file before move? {os.path.isfile(source_item_path)}")
                                        logger.info(f"      Source is dir before move? {os.path.isdir(source_item_path)}")

                                        logger.info(f"    Attempting to move {source_item_path} to {destination_item_path}")
                                        try:
                                            if os.path.exists(source_item_path): # Extra check before actual move
                                                shutil.move(source_item_path, destination_item_path)
                                                logger.info(f"      Successfully reported move of {item_name}")
                                                logger.info(f"      Destination exists after move? {os.path.exists(destination_item_path)}")
                                                logger.info(f"      Source still exists after move? {os.path.exists(source_item_path)}")
                                            else:
                                                logger.warning(f"      Source {source_item_path} disappeared before move operation for item '{item_name}'")
                                        except Exception as e:
                                            logger.error(f"      Error moving {source_item_path} to {destination_item_path}: {e}", exc_info=True)
                                            logger.info(f"      Checking paths after error: Source exists? {os.path.exists(source_item_path)}, Dest exists? {os.path.exists(destination_item_path)}")
                                    
                                    logger.info(f"  Finished iterating through items for moving from {absolute_metrics_output_dir} to {new_qps_specific_dir_path}")
                                    if os.path.exists(absolute_metrics_output_dir):
                                        logger.info(f"    Final contents of source dir '{absolute_metrics_output_dir}': {os.listdir(absolute_metrics_output_dir)}")
                                    else:
                                        logger.info(f"    Source dir '{absolute_metrics_output_dir}' no longer exists or was moved itself.")
                                    
                                    if os.path.exists(new_qps_specific_dir_path):
                                        logger.info(f"    Final contents of dest dir '{new_qps_specific_dir_path}': {os.listdir(new_qps_specific_dir_path)}")
                                    else:
                                        logger.info(f"    Dest dir '{new_qps_specific_dir_path}' does not exist.")
                                else:
                                    logger.warning(f"  Metrics output directory {absolute_metrics_output_dir} does not exist or is not a directory. Skipping telemetry tagging for vajra.")
                            else:
                                logger.warning(f"  'veeksha/' not found as a separable segment in '{relative_metrics_output_dir_from_yaml}'. Cannot use CWD-based path strategy. Skipping telemetry tagging for vajra.")
                        else:
                            logger.warning("  `metrics_config_output_dir` not found in server_config. Skipping telemetry tagging for vajra.")

                except Exception as e:
                    logger.error(f"Error killing server: {str(e)}", exc_info=True)
                    if stdout_file: stdout_file.close()
                    if stderr_file: stderr_file.close()
        
                # create and save cache visualizations
                if self.args.cache_telemetry_path is not None and new_output_cache_file is not None and new_output_cache_file_ts is not None:
                    try:
                        visualize_cache_telemetry(new_output_cache_file, new_output_cache_file_ts)
                    except Exception as e:
                        logger.error(f"Error visualizing cache telemetry: {str(e)}", exc_info=True)

        if not found_valid_qps:
            print(
                f"No valid QPS found for {self.job_config.get_human_readable_name()}",
            )
            return {}

        print(
            f"Max QPS under SLO for {self.job_config.get_human_readable_name()} - "
            f"QPS: {max_qps_under_sla}, "
            f"TBT P{self.args.tbt_percentile * 100}: {tbt_at_max_qps}, "
            f"TTFT P{self.args.ttft_percentile * 100}: {ttft_at_max_qps}, "
            f"TPOT P{self.args.tpot_percentile * 100}: {tpot_at_max_qps}, "
            f"Deadline Miss Rate P{self.args.deadline_miss_rate_percentile * 100}: {deadline_miss_rate_at_max_qps}"
            f"Best Run ID: {best_run_id}",
        )

        if self.args.wandb_project is not None and self.args.enable_wandb_sweep:
            best_run = wandb.Api().run(f"{self.args.wandb_project}/{best_run_id}")
            best_run.tags.append("BEST_CONFIG")
            best_run.update()

        return {
            **self.job_config.to_config_dict(),
            "max_qps_under_sla": max_qps_under_sla,
            "deadline_miss_rate_at_max_qps": deadline_miss_rate_at_max_qps,
        }


def visualize_cache_telemetry(cache_telemetry_path, cache_telemetry_path_ts):
    """
    Visualize cache telemetry data by calling the visualization script.
    
    Args:
        cache_telemetry_path: Path to the cache telemetry data (file or directory pattern)
    """
    import os
    import subprocess
    from pathlib import Path
    
    output_dir = f"./visualizations_{Path(cache_telemetry_path).parent.name}"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the absolute path of the visualization script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualization_script = os.path.join(script_dir, "visualize_cache_telemetry.py")
    
    # Build the command
    cmd = ["python", visualization_script, "--file", cache_telemetry_path, "--file_ts", cache_telemetry_path_ts, "--output", output_dir]
    
    print(f"Running visualization command: {' '.join(cmd)}")
    
    # Execute the visualization script
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        print(f"Cache telemetry visualizations created in {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error visualizing cache telemetry: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False