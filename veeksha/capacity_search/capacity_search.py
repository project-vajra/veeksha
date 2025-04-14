import argparse
import glob
import json
import subprocess
import os
import signal
import requests
import time
import yaml
from typing import Optional, Tuple

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
        self, qps: float
    ) -> Tuple[
        bool, Optional[float], Optional[float], Optional[float], Optional[float], str
    ]:
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
                str(self.job_config.request_generator_config.trace_file_name),
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

        # get output_cache data and tag with current qps. Rename so that it's not overwritten by next run
        cache_output_path = self.args.cache_telemetry_path
        two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_cache_file = os.path.join(two_levels_up, cache_output_path)
        
        if os.path.exists(output_cache_file):
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
            hash_key = _get_hash(overall_key)
            run_dir = os.path.join(
                self.args.output_dir,
                str(self.job_config.server_config.openai_server_engine),
                self.job_config.model_config.name,
                # f"ttft_slack_{self.args.ttft_slack_slo}_tbt_{self.args.tbt_slo}",
                str(self.job_config.request_generator_config.trace_file_name),
                f"{hash_key}_q{qps}",
            )

            cached_request_level_metrics_file = self._get_request_level_metrics(run_dir)
            print(f"Cached request level metrics file: {cached_request_level_metrics_file}")
            if cached_request_level_metrics_file is None:
                print(f"Cache for qps {qps} not found, starting server")

                # Define the command and arguments as a list
                if self.args.server_launch_file is not None:
                    print(f"Loading server config from {self.args.server_launch_file}")
                    # Load server configuration from YAML file
                    with open(self.args.server_launch_file, 'r') as f:
                        server_config = yaml.safe_load(f)
                    
                    print("Constructing server command from config")
                    # Construct command from YAML configuration
                    cmd = ["python", "-m", server_config["module"]]
                    
                    # Add all configuration parameters from YAML
                    for key, value in server_config.items():
                        if key == "module": 
                            continue
                        if key == "json_model_override_args":
                            cmd.extend(["--json-model-override-args", json.dumps(value)])
                        else:
                            param_key = key.replace("_", "-")
                            if isinstance(value, bool) and value:
                                cmd.append(f"--{param_key}")
                            elif not isinstance(value, bool):
                                cmd.extend([f"--{param_key}", str(value)])
                    port = server_config["port"]
                else:
                    raise ValueError("Server launch file not specified")

                if self.is_port_in_use(port):
                    logger.warning(f"Port {port} is already in use, attempting cleanup")
                    try:
                        # Try to find and kill process using the port
                        print(f"Running fuser -k {port}/tcp")
                        subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False)
                        # time.sleep(2)
                        if self.is_port_in_use(port):
                            logger.warning(f"Failed to free port {port}, incrementing port number")
                            port = port + 1
                    except Exception as e:
                        logger.error(f"Error freeing port: {str(e)}")
                        port = port + 1

                print(f"Final command: {' '.join(cmd)}")
                print(f"Starting server process on port {port}")

                try:
                    print(f"Opening log files in {self.args.output_dir}")
                    # Open log files for stdout and stderr
                    stdout_file = open(f"{self.args.output_dir}/server_stdout.log", "w")
                    stderr_file = open(f"{self.args.output_dir}/server_stderr.log", "w")
                    
                    # Redirect output to files
                    start_time = time.time()
                    print("Launching server subprocess")
                    server_process = subprocess.Popen(
                        cmd,
                        stdout=stdout_file,  # Redirect stdout to file
                        stderr=stderr_file,  # Redirect stderr to file
                        text=True,
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

                # wait for server to start
                print("Waiting for server startup (180s)...")
                time.sleep(180)
                print(f"Server startup wait complete after {time.time() - start_time:.1f}s")

            (
                is_under_sla,
                tbt,
                ttft,
                tpot,
                deadline_miss_rate,
                run_id,
            ) = self.is_under_sla(qps)

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

            print(f"Cached request level metrics file: {cached_request_level_metrics_file}")
            if cached_request_level_metrics_file is None:
                try:
                    print(f"Terminating server process group {pid}")
                    # Kill the entire process group (server and any child processes it spawned)
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    
                    # Give it a moment to shut down gracefully
                    time.sleep(3)
                    
                    # If it's still running, force kill
                    if server_process.poll() is None:
                        logger.warning("Server didn't terminate gracefully, sending SIGKILL")
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                        time.sleep(1)

                    stdout_file.close()
                    stderr_file.close()
                    
                    print(f"Server terminated with exit code: {server_process.returncode}")
                except Exception as e:
                    logger.error(f"Error killing server: {str(e)}", exc_info=True)
                    stdout_file.close()
                    stderr_file.close()
                    # get output_cache data and tag with current qps. Rename so that it's not overwritten by next run
                    cache_output_path = self.args.cache_telemetry_path
                    two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    output_cache_file = os.path.join(two_levels_up, cache_output_path)
                    
                    if os.path.exists(output_cache_file):
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

                # get output_cache data and tag with current qps. Rename so that it's not overwritten by next run
                cache_output_path = self.args.cache_telemetry_path
                two_levels_up = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                output_cache_file = os.path.join(two_levels_up, cache_output_path)
                
                if os.path.exists(output_cache_file):
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
        
                # create and save cache visualizations
                if self.args.cache_telemetry_path is not None:
                    visualize_cache_telemetry(self.args.cache_telemetry_path)

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


def visualize_cache_telemetry(cache_telemetry_path):
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
    

    # Use the parent directory of the cache telemetry path for the pattern
    if os.path.isfile(cache_telemetry_path) or cache_telemetry_path.endswith('.json'):
        # If it's a file, use its directory
        pattern = os.path.join(os.path.dirname(cache_telemetry_path), "*.json")
    else:
        # If it's already a directory, use it directly
        pattern = os.path.join(cache_telemetry_path, "*.json")
    
    # Build the command
    cmd = ["python", visualization_script, "--pattern", pattern, "--output", output_dir]
    
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