"""
This file contains the wrapper for the benchmarking.
It first launches OPEN AI server for vajra, and then runs -m veeksha.run_benchmark
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import time
from typing import Any, Dict, Optional, Tuple, List

import ray
import yaml
from jinja2 import Environment, FileSystemLoader
from ray.util import get_node_ip_address

from veeksha.capacity_search.config.config import (
    BenchmarkConfig,
    ClientConfig,
    JobConfig,
    ModelConfig,
    ParallelConfig,
    RequestGeneratorConfig,
    ServerConfig,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)

# Define the path for the experiment cache file
EXPERIMENT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "experiment_cache.json"
)

# Define the path for the current experiment config file
CURRENT_EXPERIMENT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "current_experiment_config.json"
)

def get_experiment_config() -> Optional[Dict[str, Any]]:
    """
    Get the global experiment configuration.
    
    Returns:
        Optional[Dict[str, Any]]: The current experiment configuration or None if not set
    """
    
    # If not available in memory, try to load from file
    if os.path.exists(CURRENT_EXPERIMENT_CONFIG_PATH):
        try:
            with open(CURRENT_EXPERIMENT_CONFIG_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading experiment config from file: {e}")
    
    return None

def set_experiment_config(config: Optional[Dict[str, Any]]) -> None:
    """
    Set the global experiment configuration.
    
    Args:
        config (Optional[Dict[str, Any]]): The experiment configuration to set globally
    """
    # Also save to file for cross-process access
    if config is not None:
        try:
            with open(CURRENT_EXPERIMENT_CONFIG_PATH, "w") as f:
                json.dump(config, f)
            logger.info(f"Saved experiment config to {CURRENT_EXPERIMENT_CONFIG_PATH}")
        except IOError as e:
            logger.error(f"Error saving experiment config to file: {e}")
    else:
        # If config is None, remove the file if it exists
        if os.path.exists(CURRENT_EXPERIMENT_CONFIG_PATH):
            try:
                os.remove(CURRENT_EXPERIMENT_CONFIG_PATH)
            except IOError as e:
                logger.error(f"Error removing experiment config file: {e}")

ResourceMapping = List[Tuple[str, int]]  # List of (node_ip, gpu_id)
ReplicaResourceMapping = Dict[str, ResourceMapping]


# Code from rayutils.py - Etalon
def get_ip() -> str:
    return socket.gethostbyname(socket.gethostname())


def get_nodes() -> List[str]:
    cluster_resources_keys = list(ray.available_resources().keys())
    ip_addresses = [
        x
        for x in cluster_resources_keys
        if x.startswith("node:") and x != "node:__internal_head__"
    ]
    return ip_addresses


def get_ready_promises(promises):
    incomplete_promises = []
    for promise in promises:
        try:
            ray.get(promise, timeout=0)
        except ray.exceptions.GetTimeoutError:
            incomplete_promises.append(promise)
        except Exception as e:
            logger.error(f"Error in promise: {e}")
    return incomplete_promises


@ray.remote
class ResourceManager:
    def __init__(self):
        self.nodes = get_nodes()
        self.num_nodes = len(self.nodes)
        self.num_total_gpus = ray.available_resources()["GPU"]

        assert self.num_nodes > 0, "No nodes found in the cluster"
        assert self.num_total_gpus > 0, "No GPUs found in the cluster"
        assert (
            self.num_total_gpus % self.num_nodes == 0
        ), f"Number of GPUs ({self.num_total_gpus}) is not divisible by the number of nodes ({self.num_nodes})"

        self.gpus_per_node = int(self.num_total_gpus // self.num_nodes)

        self.gpu_free_map = {node: [True] * self.gpus_per_node for node in self.nodes}
        self.node_free_map = {node: True for node in self.nodes}

    def get_replica_resource_mapping(
        self, num_gpus: int
    ) -> Optional[ReplicaResourceMapping]:
        """
        Assign node and gpu for a job
        Note that right now we only support single replica mapping
        """

        assert (
            num_gpus <= self.num_total_gpus
        ), f"Requested {num_gpus} GPUs, but only {self.num_total_gpus} are present in the cluster"

        is_multi_node = num_gpus > self.gpus_per_node
        if is_multi_node:
            assert (
                num_gpus % self.gpus_per_node == 0
            ), f"Number of GPUs ({num_gpus}) is not divisible by the number of GPUs per node ({self.gpus_per_node})"
            num_nodes = num_gpus // self.gpus_per_node

            num_free_nodes = sum(self.node_free_map.values())
            if num_free_nodes < num_nodes:
                return {}

            resource_mapping = []
            for node in self.nodes:
                if self.node_free_map[node]:
                    self.node_free_map[node] = False
                    for i in range(self.gpus_per_node):
                        self.gpu_free_map[node][i] = False
                        resource_mapping.append((node, i))

                    if len(resource_mapping) == num_gpus:
                        return {"0": resource_mapping}
        else:
            # all GPUs must be allocated on the same node and contiguously
            for node in self.nodes:
                resource_mapping = []
                for gpu_id, is_gpu_free in enumerate(self.gpu_free_map[node]):
                    # we don't want to allocate gpu combinations like 1,2
                    if not resource_mapping and gpu_id % num_gpus != 0:
                        continue

                    if is_gpu_free:
                        resource_mapping.append((node, gpu_id))
                    else:
                        # this ensures that we allocate contiguously
                        resource_mapping = []

                    if len(resource_mapping) == num_gpus:
                        self.node_free_map[node] = False
                        for _, i in resource_mapping:
                            self.gpu_free_map[node][i] = False
                        return {"0": resource_mapping}

        # currently we only support single replica allocation
        return {}

    def release_resources(self, replica_resource_mapping: ReplicaResourceMapping):
        for resource_mapping in replica_resource_mapping.values():
            for node, gpu_id in resource_mapping:
                self.gpu_free_map[node][gpu_id] = True

        for node in self.nodes:
            if all(self.gpu_free_map[node]):
                self.node_free_map[node] = True


class RayParallelRunner:
    def __init__(self):
        self.resource_manager = ResourceManager.remote()

    def map(self, func, job_configs):
        # try to assign a core to each task
        promises = []

        remote_func = ray.remote(func)

        job_configs_with_num_gpus = [
            (job_config, job_config.get_num_gpus()) for job_config in job_configs
        ]
        # this reduces fragmentation
        job_configs_with_num_gpus.sort(key=lambda x: x[1])

        for job_config, num_gpus in job_configs_with_num_gpus:
            replica_resource_mapping = {}
            while not replica_resource_mapping:
                # try to pop the promises so that we can get error messages
                promises = get_ready_promises(promises)

                replica_resource_mapping = ray.get(
                    self.resource_manager.get_replica_resource_mapping.remote(num_gpus)
                )
                time.sleep(0.1)
            # launch the task
            runner_node = replica_resource_mapping["0"][0][
                0
            ]  # replica 0, first worker, node
            promise = remote_func.options(resources={runner_node: 0.001}).remote(
                self.resource_manager, replica_resource_mapping, job_config
            )
            promises.append(promise)

        return ray.get(promises)


# benchmark_wrapper.py
def extract_ip(string):
    return re.findall(r"[0-9]+(?:\.[0-9]+){3}", string)[0]


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


@ray.remote
class OpenAIServerWrapper:
    def __init__(
        self, replica_resource_mapping: ReplicaResourceMapping, port: int = None
    ):
        self.process = None
        self.port = port
        self.replica_resource_mapping = replica_resource_mapping
        gpu_devices = []
        curr_node_ip = get_node_ip_address()
        for i in range(len(self.replica_resource_mapping["0"])):
            node_ip = extract_ip(self.replica_resource_mapping["0"][i][0])
            if curr_node_ip == node_ip:
                gpu_devices.append(self.replica_resource_mapping["0"][i][1])
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in gpu_devices])

    def get_openai_server_command(
        self,
        openai_server_engine=None,
        openai_server_model="gpt-3.5-turbo",
        openai_api_key=None,
        tp=1,
        pp=1,
        rope_scaling_type=None,
        rope_scaling_factor=None,
        fixed_chunk_size=512,
        min_chunk_size=1,
        max_chunk_size=512,
        schedule_policy=None,
        scheduler_config="FIXED_CHUNK",
        drafter=None,
        drafter_tokens=10,
        drafter_rope_scaling_type=None,
        drafter_rope_scaling_factor=None,
        max_model_len=32768,
        chat_template=None,
    ) -> str:
        template_dir_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "engine_templates")
        )
        env = Environment(loader=FileSystemLoader(template_dir_path))
        template = None
        cuda_devices = None

        if openai_server_engine == "vajra":
            template = env.get_template("vajra_template.jinja")
        elif openai_server_engine == "vllm":
            assert (
                openai_api_key is not None
            ), "OpenAI API key is required for vLLM engine"
            template = env.get_template("vllm_template.jinja")
        elif openai_server_engine == "sglang":
            template = env.get_template("sglang_template.jinja")

        rope_scaling = (
            f'{{"type":"{rope_scaling_type}", "factor": {rope_scaling_factor}}}'
        )

        drafter_rope_scaling = f'{{"type":"{drafter_rope_scaling_type}", "factor": {drafter_rope_scaling_factor}}}'

        data = {
            "model": {
                "identifier": openai_server_model,
                "chat_template": chat_template,
            },
            "server": {
                "openai_api_key": openai_api_key,
                "fixed_chunk_size": fixed_chunk_size,
                "min_chunk_size": min_chunk_size,
                "max_chunk_size": max_chunk_size,
                "schedule_policy": schedule_policy,
                "scheduler_config": scheduler_config,
            },
            "parallel_spec": {"tp_dimension": tp, "pp_dimension": pp},
            "port": self.port,
            "rope_scaling": rope_scaling,
            "rope_scaling_type": rope_scaling_type,
            "rope_scaling_factor": rope_scaling_factor,
            "cuda_devices": cuda_devices,
            "drafter": drafter,
            "drafter_tokens": drafter_tokens,
            "drafter_rope_scaling": drafter_rope_scaling,
            "max_model_len": max_model_len,
            "chat_template": chat_template,
        }

        cmd = template.render(data)

        print("running command: ", cmd, flush=True)

        return cmd

    def launch_openai_server(
        self,
        openai_server_engine=None,
        openai_server_model="gpt-3.5-turbo",
        openai_api_key=None,
        tp=1,
        pp=1,
        fixed_chunk_size=512,
        min_chunk_size=1,
        max_chunk_size=512,
        schedule_policy="fcfs",
        scheduler_config="FIXED_CHUNK",
        drafter=None,
        drafter_tokens=10,
        chat_template=None,
    ):
        """
        Setup the OPEN AI server
        If no engine is specified, it defaults to actual OPEN AI server itself.
        """
        openai_server_command = None
        if openai_server_engine in ["vllm", "sglang", "vajra"]:
            openai_server_command = self.get_openai_server_command(
                openai_server_engine=openai_server_engine,
                openai_server_model=openai_server_model,
                openai_api_key=openai_api_key,
                tp=tp,
                pp=pp,
                rope_scaling_type=("linear"),
                rope_scaling_factor=4.0,
                fixed_chunk_size=fixed_chunk_size,
                min_chunk_size=min_chunk_size,
                max_chunk_size=max_chunk_size,
                schedule_policy=schedule_policy,
                scheduler_config=scheduler_config,
                drafter=drafter,
                drafter_tokens=drafter_tokens,
                drafter_rope_scaling_type=("linear"),
                drafter_rope_scaling_factor=16.0,
                chat_template=chat_template,
            )
            logger.info(
                f"Launching OPEN AI server with command: {openai_server_command}"
            )
            self.process = subprocess.Popen(
                openai_server_command, shell=True, preexec_fn=os.setsid
            )

        elif openai_server_engine == "default" or openai_server_engine is None:
            # just use the actual OPEN AI server
            pass
        else:
            logger.error(f"Invalid engine: {openai_server_engine}")
            raise ValueError(f"Invalid engine: {openai_server_engine}")

    def stop_openai_server(self):
        """
        Stops the OPEN AI server
        """
        if self.process is not None:
            logger.info("Stopping OPEN AI server")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)


def setup_api_environment(
    openai_server_engine=None,
    openai_api_key=None,
    openai_port=None,
):
    # just make sure that OPENAI_API_KEY/BASE doesn't change for other ray tasks when setting for this one.
    # checked by printing, and it doesn't change
    if openai_server_engine in ["vllm", "sglang", "vajra", "default"]:
        if openai_server_engine in ["vllm"]:
            assert (
                openai_api_key is not None
            ), "OpenAI API key is required for VLLM engine"
        assert openai_port is not None, "OpenAI port is required"
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["OPENAI_API_BASE"] = f"http://localhost:{openai_port}/v1"


def is_default_engine(engine) -> bool:
    return engine == "default" or engine is None


def get_default_config(engine_name: str) -> dict:
    """
    Fetch the default configuration for a specified server engine.

    Args:
        engine_name (str): Name of the server engine (e.g., 'vajra', 'vllm', 'sglang')

    Returns:
        dict: Default configuration for the specified engine
    """
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "config_editor/default_engine_configs",
        f"{engine_name}_config.yml",
    )

    try:
        with open(config_path, "r") as f:
            default_config = yaml.safe_load(f)
        return default_config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Default configuration file for engine '{engine_name}' not found at {config_path}"
        )
    except yaml.YAMLError as e:
        raise ValueError(
            f"Error parsing default configuration file for engine '{engine_name}': {e}"
        )


def run_from_config(config_path: str):
    """
    Reads a YAML configuration file containing a single test combination,
    generates JobConfig, BenchmarkConfig, and ParallelConfig objects, and runs the benchmark.

    Args:
        config_path (str): Path to the configuration YAML file.
    """
    # Load YAML config
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Check if this experiment is already in the cache
    if "metadata" in config and "config_id" in config["metadata"]:
        config_id = config["metadata"]["config_id"]
        if is_experiment_in_cache(config_id):
            logger.info(
                f"Experiment with config_id {config_id} already completed, skipping"
            )
            return
        logger.info(f"Running experiment with config_id {config_id}")
    else:
        logger.warning(
            f"Config file {config_path} does not have a config_id, cannot use cache"
        )

    # Fetch the default engine configuration so that we don't have missing keys
    default_config = get_default_config(config["server"]["openai_server_engine"])

    # Extract items from config with fallback to defaults
    model = config.get("model", default_config.get("model", {}))
    request_generator = config.get(
        "request_generator_config", default_config.get("request_generator_config", {})
    )
    request_config = config.get(
        "request_config", default_config.get("request_config", {})
    )
    server = config.get("server", default_config.get("server", {}))
    benchmark_conf = config.get(
        "benchmark_config", default_config.get("benchmark_config", {})
    )
    parallel_spec = config.get("parallel_spec", default_config.get("parallel_spec", {}))

    # Log which sections are using provided vs default values
    for section in [
        "model",
        "request_generator_config",
        "request_config",
        "server",
        "benchmark_config",
        "parallel_spec",
    ]:
        if section in config:
            logger.info(f"Using provided section for {section}")
        elif section in default_config:
            logger.info(f"Using default section for {section}")

    # Create configuration objects
    parallel_config = ParallelConfig(
        tensor_parallel_size=parallel_spec.get("tp_dimension", 1),
        pipeline_parallel_size=parallel_spec.get("pp_dimension", 1),
    )

    model_config = ModelConfig(
        name=model.get("name", None),
        identifier=model.get("identifier", None),
    )

    chat_template = model.get("chat_template", "")

    # Build request generator config using the extracted request_generator dictionary
    request_generator_config = RequestGeneratorConfig(**request_generator)
    logger.info(f"Final request generator config: {request_generator}")

    # Build client config using the extracted request_config dictionary
    client_fields = [
        "num_clients",
        "num_concurrent_requests_per_client",
        "timeout",
        "max_num_completed_requests",
        "additional_sampling_params",
        "llm_api",
    ]

    client_config_dict = {
        field: request_config.get(field)
        for field in client_fields
        if field in request_config
    }
    client_config = ClientConfig(**client_config_dict)
    logger.info(f"Final client config: {client_config_dict}")

    openai_port = server.get(
        "openai_api_port", 8000
    )  # Default to 8000 if not specified

    # Build server config using the extracted server dictionary
    server_fields = [
        "openai_server_engine",
        "openai_api_url",
        "openai_api_key",
        "fixed_chunk_size",
        "min_chunk_size",
        "max_chunk_size",
        "schedule_policy",
        "scheduler_config",
    ]

    server_config_dict = {
        field: server.get(field) for field in server_fields if field in server
    }
    server_config = ServerConfig(**server_config_dict)
    logger.info(f"Final server config: {server_config_dict}")

    job_config = JobConfig(
        model_config=model_config,
        request_generator_config=request_generator_config,
        client_config=client_config,
        server_config=server_config,
    )

    # Build benchmark config using the extracted benchmark_conf dictionary
    benchmark_fields = ["output_dir", "qps", "should_use_given_dir"]

    benchmark_config_dict = {
        field: benchmark_conf.get(field)
        for field in benchmark_fields
        if field in benchmark_conf
    }
    benchmark_config = BenchmarkConfig(**benchmark_config_dict)
    logger.info(f"Final benchmark config: {benchmark_config_dict}")

    # Creating the ResourceManager actor to get GPU resources
    resource_manager = ResourceManager.remote()
    num_gpus = (
        parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
    )

    # Retry until resource is available
    replica_resource_mapping = {}
    while not replica_resource_mapping:
        replica_resource_mapping = ray.get(
            resource_manager.get_replica_resource_mapping.remote(num_gpus)
        )
        if not replica_resource_mapping:
            print("Waiting for GPU resources to be free...", flush=True)
            time.sleep(5)

    # Set the global experiment configuration
    set_experiment_config(config)

    # Run the benchmark
    success = run(
        job_config,
        benchmark_config,
        replica_resource_mapping,
        openai_port,
        parallel_config,
        chat_template,
    )

    # If the experiment ran successfully and has a config_id, add it to the cache
    if success and "metadata" in config and "config_id" in config["metadata"]:
        add_experiment_to_cache(config["metadata"]["config_id"])
    
    # Clean up the current experiment config file
    if os.path.exists(CURRENT_EXPERIMENT_CONFIG_PATH):
        try:
            os.remove(CURRENT_EXPERIMENT_CONFIG_PATH)
        except IOError as e:
            logger.error(f"Error deleting current experiment config file: {e}")


def run(
    job_config: JobConfig,
    benchmark_config: BenchmarkConfig,
    replica_resource_mapping: ReplicaResourceMapping,
    openai_port: int,
    parallel_config: ParallelConfig,
    chat_template: str,
):
    """
    Main function

    Returns:
        bool: True if the benchmark completed successfully, False otherwise
    """

    num_gpus = (
        parallel_config.tensor_parallel_size * parallel_config.pipeline_parallel_size
        if not is_default_engine(job_config.server_config.openai_server_engine)
        else 0
    )

    # If server runs across multiple nodes, then just create set of nodes and pass it to resources parameter for Ray Actor
    set_of_nodes = set(
        replica_resource_mapping["0"][i][0]
        for i in range(len(replica_resource_mapping["0"]))
    )
    resources = {i: 0.001 for i in set_of_nodes}

    openai_server_wrapper = OpenAIServerWrapper.options(
        num_gpus=num_gpus, resources=resources
    ).remote(replica_resource_mapping=replica_resource_mapping, port=openai_port)

    print(
        f"\n\nOPEN AI SERVER {job_config.server_config.openai_server_engine} PORT: {openai_port}\n\n",
        flush=True,
    )

    try:
        # Launch the OPEN AI server
        ray.get(
            openai_server_wrapper.launch_openai_server.remote(
                openai_server_engine=job_config.server_config.openai_server_engine,
                openai_server_model=job_config.model_config.identifier,
                openai_api_key=job_config.server_config.openai_api_key,
                tp=parallel_config.tensor_parallel_size,
                pp=parallel_config.pipeline_parallel_size,
                fixed_chunk_size=job_config.server_config.fixed_chunk_size,
                min_chunk_size=job_config.server_config.min_chunk_size,
                max_chunk_size=job_config.server_config.max_chunk_size,
                schedule_policy=job_config.server_config.schedule_policy,
                scheduler_config=job_config.server_config.scheduler_config,
                chat_template=chat_template,
            )
        )

        setup_api_environment(
            openai_server_engine=job_config.server_config.openai_server_engine,
            openai_api_key=job_config.server_config.openai_api_key,
            openai_port=openai_port,
        )

        # Wait for the server to start. For 70B model, it takes around 2 minutes to start
        sleep_time = (
            0
            if is_default_engine(job_config.server_config.openai_server_engine)
            else 60
        )
        time.sleep(sleep_time)

        # Additional retry mechanism to check if server is up
        count = 0
        while not is_port_in_use(openai_port):
            logger.info(
                f"Waiting for OPEN AI server to start. Port {openai_port} is not in use"
            )
            time.sleep(60)
            if count > 1:
                raise RuntimeError("OPEN AI server did not start after 2 minutes.")
            count += 1

        # Run the benchmark
        benchmark_command = f"python -m veeksha.run_benchmark {job_config.to_args()} {benchmark_config.to_args()}"

        print(benchmark_command)

        benchmark_process = subprocess.Popen(benchmark_command, shell=True)
        benchmark_process.wait()

        # Check if the benchmark process completed successfully
        if benchmark_process.returncode != 0:
            logger.error(
                f"Benchmark process exited with non-zero return code: {benchmark_process.returncode}"
            )
            return False

        return True
    except Exception as e:
        logger.error(f"Error during benchmark execution: {str(e)}")
        return False
    finally:
        # Stop the OPEN AI server regardless of success or failure
        try:
            ray.get(openai_server_wrapper.stop_openai_server.remote())
        except Exception as e:
            logger.error(f"Error stopping OPEN AI server: {str(e)}")

        ray.shutdown()


def server_benchmark_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Path to a single experiment config YAML"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        help="Paths to multiple experiment config YAMLs to run sequentially",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        help="Path to directory containing experiment config YAMLs to run sequentially",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the experiment cache before running",
    )
    args = parser.parse_args()

    # Clear the cache if requested
    if args.clear_cache:
        if os.path.exists(EXPERIMENT_CACHE_PATH):
            os.remove(EXPERIMENT_CACHE_PATH)
            logger.info(f"Cleared experiment cache at {EXPERIMENT_CACHE_PATH}")
        else:
            logger.info("No experiment cache to clear")

    # Validate arguments
    if (
        sum(arg is not None for arg in [args.config, args.configs, args.config_dir])
        != 1
    ):
        parser.error(
            "Exactly one of --config, --configs, or --config-dir must be specified"
        )

    # Collect all config paths to run
    config_paths = []

    if args.config:
        config_paths = [args.config]
    elif args.configs:
        config_paths = args.configs
    elif args.config_dir:
        config_dir = args.config_dir
        if not os.path.isdir(config_dir):
            parser.error(f"Config directory {config_dir} does not exist")
        # Get all YAML files in the directory
        config_paths = [
            os.path.join(config_dir, f)
            for f in os.listdir(config_dir)
            if f.endswith(".yml") or f.endswith(".yaml")
        ]
        if not config_paths:
            parser.error(f"No YAML files found in {config_dir}")
        # Sort for deterministic ordering
        config_paths.sort()

    # Load the experiment cache
    cache = load_experiment_cache()
    logger.info(
        f"Loaded experiment cache with {len(cache['completed_experiments'])} completed experiments"
    )

    # Run all configurations sequentially
    logger.info(
        f"Preparing to run {len(config_paths)} benchmark configurations sequentially"
    )

    completed_count = 0
    skipped_count = 0
    failed_count = 0

    for i, config_path in enumerate(config_paths):
        logger.info(f"\n\n")
        logger.info(f"---------- Processing benchmark {i+1}/{len(config_paths)}: {config_path} ----------")

        # Check if this experiment is in the cache before loading the full config
        try:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)

            if "metadata" in config and "config_id" in config["metadata"]:
                config_id = config["metadata"]["config_id"]
                if is_experiment_in_cache(config_id):
                    logger.info(
                        f"Experiment with config_id {config_id} already completed, skipping"
                    )
                    skipped_count += 1
                    continue
        except Exception as e:
            logger.error(f"Error checking cache for {config_path}: {str(e)}")
            # Continue with normal execution if we can't check the cache

        try:
            run_from_config(config_path)
            logger.info(
                f"Successfully completed benchmark {i+1}/{len(config_paths)}: {config_path}"
            )
            completed_count += 1
        except Exception as e:
            logger.error(
                f"Error running benchmark {i+1}/{len(config_paths)}: {config_path}"
            )
            logger.error(f"Error details: {str(e)}")
            logger.error("Continuing with next benchmark...")
            failed_count += 1

    logger.info(
        f"Benchmark summary: {completed_count} completed, {skipped_count} skipped (cached), {failed_count} failed"
    )
    logger.info(f"All {len(config_paths)} benchmarks processed")
    
    # Clean up the current experiment config file at the end of all experiments
    if os.path.exists(CURRENT_EXPERIMENT_CONFIG_PATH):
        try:
            os.remove(CURRENT_EXPERIMENT_CONFIG_PATH)
        except IOError as e:
            logger.error(f"Error deleting current experiment config file: {e}")


# Cache management functions
def load_experiment_cache():
    """
    Load the experiment cache from disk.

    Returns:
        dict: A dictionary of completed experiment config_ids
    """
    if not os.path.exists(EXPERIMENT_CACHE_PATH):
        return {"completed_experiments": []}

    try:
        with open(EXPERIMENT_CACHE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        logger.warning(
            f"Could not load experiment cache from {EXPERIMENT_CACHE_PATH}, creating new cache"
        )
        return {"completed_experiments": []}


def save_experiment_cache(cache):
    """
    Save the experiment cache to disk.

    Args:
        cache (dict): The cache dictionary to save
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(EXPERIMENT_CACHE_PATH), exist_ok=True)

    with open(EXPERIMENT_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def is_experiment_in_cache(config_id):
    """
    Check if an experiment with the given config_id is in the cache.

    Args:
        config_id (str): The config_id to check

    Returns:
        bool: True if the experiment is in the cache, False otherwise
    """
    cache = load_experiment_cache()
    return config_id in cache["completed_experiments"]


def add_experiment_to_cache(config_id):
    """
    Add an experiment with the given config_id to the cache.

    Args:
        config_id (str): The config_id to add
    """
    cache = load_experiment_cache()
    if config_id not in cache["completed_experiments"]:
        cache["completed_experiments"].append(config_id)
        save_experiment_cache(cache)
        logger.info(f"Added experiment with config_id {config_id} to cache")
