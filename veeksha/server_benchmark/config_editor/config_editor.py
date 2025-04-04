import json
import os
import re
import sys
from pathlib import Path

import yaml

# Import cache functions from server_benchmark
try:
    from veeksha.server_benchmark.server_benchmark import (
        EXPERIMENT_CACHE_PATH,
        is_experiment_in_cache,
        load_experiment_cache,
    )

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    # Fallback implementations if import fails
    EXPERIMENT_CACHE_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiment_cache.json",
    )

    def load_experiment_cache():
        if not os.path.exists(EXPERIMENT_CACHE_PATH):
            return {"completed_experiments": []}
        try:
            with open(EXPERIMENT_CACHE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {"completed_experiments": []}

    def is_experiment_in_cache(config_id):
        cache = load_experiment_cache()
        return config_id in cache["completed_experiments"]


# For single keypress detection
try:
    import msvcrt  # Windows

    def getch():
        return msvcrt.getch().decode("utf-8")

except ImportError:
    try:
        import termios
        import tty

        # Unix/Linux/MacOS
        def unix_getch():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        getch = unix_getch
    except ImportError:
        # Fallback if neither method works
        def input_getch():
            return input()

        getch = input_getch

# --- Configuration ---
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "default_engine_configs",
    "vajra_config.yml",
)
OPTIONS_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config_options.yml"
)
MODEL_MAPPING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model_mapping.yml"
)
OUTPUT_DIR = Path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiment_configs")
)  # Where generated .yml config files are saved
BENCHMARK_BASE_DIR = (
    "results"  # Base directory for benchmark output specified in the config
)
CONFIG_EXTENSIONS = (".yml",)

# Engine parameter compatibility mapping
ENGINE_PARAM_COMPATIBILITY = {
    "vajra": {
        "server": [
            "openai_server_engine",
            "openai_api_url",
            "schedule_policy",
            "scheduler_config",
            "fixed_chunk_size",
            "min_chunk_size",
            "max_chunk_size",
            "openai_api_key",
            "openai_api_port",
        ],
        "model": ["name", "identifier", "chat_template"],
        "parallel_spec": ["tp_dimension", "pp_dimension"],
    },
    "vllm": {
        "server": [
            "openai_server_engine",
            "openai_api_url",
            "schedule_policy",
            "fixed_chunk_size",
            "openai_api_key",
            "openai_api_port",
        ],
        "model": ["name", "identifier", "chat_template"],
        "parallel_spec": ["tp_dimension", "pp_dimension"],
    },
    "sglang": {
        "server": [
            "openai_server_engine",
            "openai_api_url",
            "openai_api_key",
            "openai_api_port",
        ],
        "model": ["name", "identifier"],
        "parallel_spec": ["tp_dimension"],
    },
}

# Global variables
CONFIG_OPTIONS = None
MODEL_MAPPING = None
# Special marker for multi-value fields
MULTI_VALUE_MARKER = "__MULTI_VALUES__"


# --- Custom YAML Dumper ---
class ForceLiteralDumper(yaml.SafeDumper):
    """Custom YAML Dumper for multi-line strings using literal style ('|')."""

    def represent_scalar(self, tag, value, style=None):
        if isinstance(value, str) and "\n" in value:
            style = "|"
        if value is None:
            return super().represent_scalar(
                tag, value, style=style if style is not None else ""
            )
        return super().represent_scalar(tag, value, style=style)


# --- Helper Functions ---
def clear_screen():
    """Clears the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def show_message(message):
    """Prints a message to the console."""
    print(message)


def load_yaml_file(filepath, file_description="config", encoding="utf-8"):
    """Loads a YAML file safely."""
    try:
        filepath = Path(filepath)
        if not filepath.is_file():
            show_message(
                f"Error: {file_description.capitalize()} file not found at '{filepath}'"
            )
            return None
        with open(filepath, "r", encoding=encoding) as f:
            data = yaml.safe_load(f)
            return data if data is not None else {}
    except yaml.YAMLError as e:
        show_message(f"Error loading YAML {file_description} file '{filepath}': {e}")
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark
            show_message(
                f"  Error position: (Line: {mark.line+1}, Column: {mark.column+1})"
            )
        return None
    except Exception as e:
        show_message(f"Error loading {file_description} file '{filepath}': {e}")
        return None


def transform_config_for_saving(config_data):
    """
    Transforms the configuration data before saving to handle multi-valued fields.
    Creates cartesian combinations of all multi-valued fields, generating separate
    configurations for each valid combination.

    When a parameter has multiple values, we create separate config files for each value.
    When multiple parameters have multiple values, we create the cartesian product of all combinations.
    Each combination is filtered to only include parameters compatible with the engine.

    Returns:
        A list of transformed configurations, each representing a valid combination
    """
    if not config_data or not isinstance(config_data, dict):
        return [config_data]

    # First, identify all multi-valued fields and their values
    multi_value_fields = {}
    engines = []

    # Check if we have multiple engines
    if "server" in config_data and "openai_server_engine" in config_data["server"]:
        engine_value = config_data["server"]["openai_server_engine"]
        if is_multi_value(engine_value):
            engines = get_multi_values(engine_value)
        else:
            engines = [engine_value]

    # Collect all multi-valued fields
    for section_key, section_data in config_data.items():
        if not isinstance(section_data, dict):
            continue

        for field_key, field_value in section_data.items():
            if is_multi_value(field_value):
                if section_key not in multi_value_fields:
                    multi_value_fields[section_key] = {}
                multi_value_fields[section_key][field_key] = get_multi_values(
                    field_value
                )

    # If no multi-valued fields, just return the original config
    if not multi_value_fields:
        # Add a unique config ID for caching
        import hashlib
        import json

        # Create a deep copy of the config to avoid modifying the original
        config_copy = {}
        for section_key, section_data in config_data.items():
            if isinstance(section_data, dict):
                config_copy[section_key] = section_data.copy()
            else:
                config_copy[section_key] = section_data

        # Generate a unique ID based on the config content
        config_copy_for_hash = json.loads(json.dumps(config_copy))
        # Remove output_dir from benchmark_config to avoid hash changes when only the output directory changes
        if "benchmark_config" in config_copy_for_hash and isinstance(config_copy_for_hash["benchmark_config"], dict):
            if "output_dir" in config_copy_for_hash["benchmark_config"]:
                del config_copy_for_hash["benchmark_config"]["output_dir"]
        
        config_str = json.dumps(config_copy_for_hash, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        config_id = hash_obj.hexdigest()

        # Add the config ID to the config
        if "metadata" not in config_copy:
            config_copy["metadata"] = {}
        config_copy["metadata"]["config_id"] = config_id

        return [config_copy]

    # Generate all possible combinations
    # Start with a single empty combination
    combinations = [{}]
    
    # Process each section with multi-valued fields
    for section_key, fields in multi_value_fields.items():
        # For each section, we'll create new combinations
        new_combinations = []
        
        # For each existing partial combination
        for combo in combinations:
            # Create temporary lists to store field values for this section
            section_combinations = [{}]
            
            # Process each multi-valued field in this section
            for field_key, values in fields.items():
                temp_combinations = []
                
                # For each existing partial section combination
                for sec_combo in section_combinations:
                    # For each value of this field
                    for value in values:
                        # Create a new section combination with this value
                        new_sec_combo = sec_combo.copy()
                        new_sec_combo[field_key] = value
                        temp_combinations.append(new_sec_combo)
                
                # Update section_combinations with the new combinations
                section_combinations = temp_combinations
            
            # Now combine each section combination with the existing partial combination
            for sec_combo in section_combinations:
                new_full_combo = combo.copy()
                new_full_combo[section_key] = sec_combo
                new_combinations.append(new_full_combo)
        
        # Update the combinations list with the new combinations
        combinations = new_combinations

    # Now we have all possible combinations of multi-valued fields
    # We need to merge each combination with the original config
    result_configs = []

    # Import modules for generating unique IDs
    import hashlib
    import json

    for combo in combinations:
        # Start with a deep copy of the original config
        config_copy = {}
        for section_key, section_data in config_data.items():
            if isinstance(section_data, dict):
                config_copy[section_key] = section_data.copy()
            else:
                config_copy[section_key] = section_data

        # Apply the combination values to the config copy
        for section_key, fields in combo.items():
            for field_key, value in fields.items():
                config_copy[section_key][field_key] = value

        # Determine the engine for this combination
        engine = None
        if "server" in config_copy and "openai_server_engine" in config_copy["server"]:
            engine = config_copy["server"]["openai_server_engine"]

        # Check if this combination has valid parallel dimensions for the engine
        if engine and not has_valid_parallel_dimensions(config_copy, engine):
            # Skip combinations with invalid parallel dimensions
            continue

        # Filter parameters for this engine
        if engine:
            for section_key in ["server", "model", "parallel_spec"]:
                if section_key in config_copy and isinstance(
                    config_copy[section_key], dict
                ):
                    config_copy[section_key] = filter_params_for_engine(
                        section_key, config_copy[section_key], engine
                    )

        # Generate a unique ID based on the config content
        config_copy_for_hash = json.loads(json.dumps(config_copy))
        # Remove output_dir from benchmark_config to avoid hash changes when only the output directory changes
        if "benchmark_config" in config_copy_for_hash and isinstance(config_copy_for_hash["benchmark_config"], dict):
            if "output_dir" in config_copy_for_hash["benchmark_config"]:
                del config_copy_for_hash["benchmark_config"]["output_dir"]
                
        config_str = json.dumps(config_copy_for_hash, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        config_id = hash_obj.hexdigest()

        # Add the config ID to the config
        if "metadata" not in config_copy:
            config_copy["metadata"] = {}
        config_copy["metadata"]["config_id"] = config_id

        result_configs.append(config_copy)

    return result_configs


def has_valid_parallel_dimensions(config_data, engine):
    """
    Checks if the parallel dimensions in the config are valid for the given engine.
    This is a critical check that can invalidate a configuration.

    Args:
        config_data: The configuration data dictionary
        engine: The engine to check compatibility with

    Returns:
        Boolean indicating if the parallel dimensions are valid
    """
    if engine not in ENGINE_PARAM_COMPATIBILITY:
        return True  # If engine not found in compatibility mapping, assume compatible

    # Check parallel_spec compatibility
    if "parallel_spec" in config_data and isinstance(
        config_data["parallel_spec"], dict
    ):
        parallel_spec = config_data["parallel_spec"]

        # Check pp_dimension compatibility
        if "pp_dimension" in parallel_spec:
            pp_value = parallel_spec["pp_dimension"]
            # If pp > 1 but engine doesn't support pipeline parallelism
            if isinstance(pp_value, (int, float)) and pp_value > 1:
                if "parallel_spec" in ENGINE_PARAM_COMPATIBILITY[engine]:
                    if (
                        "pp_dimension"
                        not in ENGINE_PARAM_COMPATIBILITY[engine]["parallel_spec"]
                    ):
                        return False

        # Check tp_dimension compatibility
        if "tp_dimension" in parallel_spec:
            tp_value = parallel_spec["tp_dimension"]
            # If tp > 1 but engine doesn't support tensor parallelism
            if isinstance(tp_value, (int, float)) and tp_value > 1:
                if "parallel_spec" in ENGINE_PARAM_COMPATIBILITY[engine]:
                    if (
                        "tp_dimension"
                        not in ENGINE_PARAM_COMPATIBILITY[engine]["parallel_spec"]
                    ):
                        return False

    return True


def save_config(config_data, filename, encoding="utf-8", experiment_name=""):
    """Saves configuration data to YAML file(s) using the custom dumper."""
    if not filename:
        show_message("Error: Filename cannot be empty.")
        return False, None
    if not filename.lower().endswith(".yml"):
        if filename.lower().endswith(".yaml"):
            filename = filename[:-5]
        filename += ".yml"

    # Validate configuration compatibility
    is_valid, warnings = validate_config_compatibility(config_data)
    if not is_valid:
        clear_screen()
        print("--- Config warnings ---")
        print("The following incompatibilities were detected in the spec:")
        for warning in warnings:
            print(f"  - {warning}")
        print("\nThere won't be generated configs for the incompatible params.")
        choice = get_single_key("Save anyway? (Y/n): ", ["y", "n", "\n", "\r"])
        if choice == "\n" or choice == "\r" or choice == "y":
            pass
        else:
            show_message("Save cancelled due to configuration incompatibilities.")
            return False, None

    # Transform the config data to handle multi-valued fields
    transformed_configs = transform_config_for_saving(config_data)

    if not transformed_configs:
        show_message(
            "Error: No valid configurations generated after filtering incompatible combinations."
        )
        return False, None

    # Create a unique folder for this set of configs
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract base name without extension for folder name
    base_name = filename.rsplit(".", 1)[0]

    # Clean up the base name - remove any MULTI_VALUES markers for the folder name
    clean_base_name = base_name
    if "MULTI_VALUES" in clean_base_name:
        # Create a cleaner base name for the folder
        engine_part = ""
        model_part = ""
        pp_part = ""
        tp_part = ""
        trace_part = ""
        qps_part = ""

        # Extract key components from the config
        if "server" in config_data and "openai_server_engine" in config_data["server"]:
            if is_multi_value(config_data["server"]["openai_server_engine"]):
                engine_part = "multi-engine"
            else:
                engine_part = config_data["server"]["openai_server_engine"]

        if "model" in config_data and "identifier" in config_data["model"]:
            if is_multi_value(config_data["model"]["identifier"]):
                model_part = "multi-model"
            else:
                model_part = config_data["model"]["identifier"]
                # Extract just the model name without path
                if "/" in model_part:
                    model_part = model_part.split("/")[-1]

        if "parallel_spec" in config_data:
            if "tp_dimension" in config_data["parallel_spec"]:
                tp = config_data["parallel_spec"]["tp_dimension"]
                # Only add tp if it's not the default value of 1
                if tp != 1:
                    tp_part = str(tp)
            else:
                tp_part = "1"  # Default

            if "pp_dimension" in config_data["parallel_spec"]:
                pp = config_data["parallel_spec"]["pp_dimension"]
                # Only add pp if it's not the default value of 1
                if pp != 1:
                    pp_part = str(pp)
            else:
                pp_part = "1"  # Default

        if (
            "request_generator_config" in config_data
            and "generator_type" in config_data["request_generator_config"]
        ):
            trace_part = config_data["request_generator_config"]["generator_type"]

        if (
            "benchmark_config" in config_data
            and "qps" in config_data["benchmark_config"]
        ):
            qps_value = config_data["benchmark_config"]["qps"]
            if isinstance(qps_value, (int, float)):
                qps_part = f"qps{qps_value}"

        # Construct a clean folder name
        parts = []
        if engine_part:
            parts.append(engine_part)
        if model_part:
            parts.append(model_part)
        if tp_part and tp_part != "1":
            parts.append(f"tp{tp_part}")
        if pp_part and pp_part != "1":
            parts.append(f"pp{pp_part}")
        if trace_part:
            parts.append(trace_part)
        if qps_part:
            parts.append(qps_part)

        if parts:
            clean_base_name = "_".join(parts)

    # Add experiment name if provided
    if experiment_name:
        folder_name = f"{timestamp}_{experiment_name}"
    else:
        folder_name = f"{timestamp}_{clean_base_name}"

    # Sanitize the folder name
    folder_name = sanitize_for_filename(folder_name)

    # Create the experiment directory
    experiment_dir = OUTPUT_DIR
    folder_path = os.path.join(experiment_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Load the experiment cache to check which configs have been run
    cache_loaded = False
    completed_experiments = []
    if CACHE_AVAILABLE:
        try:
            cache = load_experiment_cache()
            completed_experiments = cache.get("completed_experiments", [])
            cache_loaded = True
        except Exception as e:
            show_message(f"Warning: Could not load experiment cache: {e}")

    # Group configurations by engine
    configs_by_engine = {}
    for config in transformed_configs:
        if "server" in config and "openai_server_engine" in config["server"]:
            engine = config["server"]["openai_server_engine"]
            if engine not in configs_by_engine:
                configs_by_engine[engine] = []
            configs_by_engine[engine].append(config)
        else:
            # Handle configs without engine specified (unlikely)
            if "unknown" not in configs_by_engine:
                configs_by_engine["unknown"] = []
            configs_by_engine["unknown"].append(config)

    # Save each engine's configs in its own subdirectory
    saved_filepaths = []
    engines_with_configs = []

    for engine, configs in configs_by_engine.items():
        # Create engine-specific subdirectory
        engine_folder_path = os.path.join(folder_path, sanitize_for_filename(engine))
        os.makedirs(engine_folder_path, exist_ok=True)
        engines_with_configs.append(engine)
        
        for i, config in enumerate(configs):
            # Generate a suffix for this specific config
            suffix = generate_config_suffix(config)

            # For single configs in an engine, use a simpler naming scheme
            if len(configs) == 1:
                config_filename = f"{engine}.yml"
            else:
                # For multiple configs, use a naming scheme based on the engine and other parameters
                if suffix:
                    config_filename = f"config_{suffix}.yml"
                else:
                    config_filename = f"config_{i+1}.yml"

            # Full path to the config file
            config_filepath = os.path.join(engine_folder_path, config_filename)

            try:
                with open(config_filepath, "w", encoding=encoding) as file:
                    yaml.dump(
                        config,
                        file,
                        Dumper=ForceLiteralDumper,
                        default_flow_style=False,
                        sort_keys=False,
                    )

                saved_filepaths.append(Path(config_filepath))
            except Exception as e:
                show_message(f"Error saving config to '{config_filepath}': {e}")
                # Continue with other configs even if one fails

    if saved_filepaths:
        # Check which configs have been run already
        cache_status = []
        for filepath in saved_filepaths:
            try:
                with open(filepath, "r") as f:
                    config = yaml.safe_load(f)

                if "metadata" in config and "config_id" in config["metadata"]:
                    config_id = config["metadata"]["config_id"]
                    if cache_loaded and is_experiment_in_cache(config_id):
                        cache_status.append(True)  # Already run
                    else:
                        cache_status.append(False)  # Not run yet
                else:
                    cache_status.append(None)  # No config_id
            except Exception:
                cache_status.append(None)  # Error reading file

        # Display the saved configs with cache status
        show_message(f"Generated {len(saved_filepaths)} config files:")
        for i, (filepath, is_cached) in enumerate(
            zip(saved_filepaths, cache_status), 1
        ):
            cache_indicator = ""
            if is_cached is True:
                cache_indicator = " [ALREADY RUN]"
            elif is_cached is False:
                cache_indicator = " [NEW]"
            show_message(f"  {i}. {filepath}{cache_indicator}")

        # Extract and display the directory path
        if saved_filepaths:
            main_dir_path = saved_filepaths[0].parent.parent
            clear_screen()
            # check cache status
            cached_count = 0
            if cache_loaded:
                new_count = sum(1 for status in cache_status if status is False)
                cached_count = sum(1 for status in cache_status if status is True)
            print("=" * 80)

            if cached_count > 0:        
                print(f"CONFIG FILES SAVED WITH PARTIAL OR TOTAL CACHE HITS")
            else:
                print(f"CONFIG FILES SAVED")

            print("=" * 80)
            print(f"Number of config files: {len(saved_filepaths)}")
            print(f"Configs organized by engine: {', '.join(engines_with_configs)}")

            # Show cache status summary if available
            if cached_count > 0:
                print(
                    f"\nCache status: {new_count} new, {cached_count} already run"
                )

            # Display the CLI commands to run the experiments
            relative_main_dir = os.path.relpath(main_dir_path)
            print("\nRun experiments with these commands (one environment per engine):")
            print("-" * 60)
            
            # Collect commands to save to a file
            command_lines = []
            command_lines.append("# Experiment run commands")
            command_lines.append("# Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            command_lines.append("")
            
            # Display command for each engine directory
            for engine in engines_with_configs:
                engine_dir = os.path.join(relative_main_dir, sanitize_for_filename(engine))
                engine_configs = [p for p in saved_filepaths if sanitize_for_filename(engine) in str(p)]
                if len(engine_configs) == 1:
                    # For a single config, show the direct config command
                    relative_path = os.path.relpath(engine_configs[0])
                    print(f"# For {engine} engine:")
                    command = f"python -m veeksha.server_benchmark --config {relative_path}"
                    print(command)
                    
                    # Add to command lines
                    command_lines.append(f"# For {engine} engine:")
                    command_lines.append(command)
                else:
                    # For multiple configs, show the config-dir command
                    print(f"# For {engine} engine:")
                    command = f"python -m veeksha.server_benchmark --config-dir {engine_dir}"
                    print(command)
                    
                    # Add to command lines
                    command_lines.append(f"# For {engine} engine:")
                    command_lines.append(command)
                print()
                command_lines.append("")
                
            print("-" * 60)
            
            # Save commands to a file in the experiment directory
            commands_filepath = os.path.join(folder_path, "run_commands.txt")
            try:
                with open(commands_filepath, "w", encoding="utf-8") as f:
                    f.write("\n".join(command_lines))
                print(f"\nRun commands saved to: {commands_filepath}")
            except Exception as e:
                print(f"Error saving commands to file: {e}")

            print("\nPress any key to continue...")
            getch()
        return True, saved_filepaths
    return False, None


def generate_config_suffix(config):
    """
    Generates a suffix for a config filename based on its unique properties.
    Only includes values that are actually set in the config file.

    Args:
        config: The configuration data dictionary

    Returns:
        A string to use as a filename suffix
    """
    parts = []

    # Add engine
    if "server" in config and "openai_server_engine" in config["server"]:
        engine = config["server"]["openai_server_engine"]
        parts.append(sanitize_for_filename(engine))

    # Add parallel dimensions only if they're actually set
    if "parallel_spec" in config:
        if "tp_dimension" in config["parallel_spec"]:
            tp = config["parallel_spec"]["tp_dimension"]
            # Only add tp if it's not the default value of 1
            if tp != 1:
                parts.append(f"tp{sanitize_for_filename(str(tp))}")
        if "pp_dimension" in config["parallel_spec"]:
            pp = config["parallel_spec"]["pp_dimension"]
            # Only add pp if it's not the default value of 1
            if pp != 1:
                parts.append(f"pp{sanitize_for_filename(str(pp))}")
    
    # Add trace file information if present
    if "request_generator_config" in config:
        req_gen_config = config["request_generator_config"]
        if "request_length_generator_provider" in req_gen_config and req_gen_config["request_length_generator_provider"] == "trace":
            if "trace_request_length_generator_trace_file" in req_gen_config:
                trace_file = req_gen_config["trace_request_length_generator_trace_file"]
                if trace_file:
                    # Extract just the base name without path and extension
                    if isinstance(trace_file, str):
                        try:
                            # For special trace names like 'decode' or 'prefill'
                            if trace_file in ["decode", "prefill"]:
                                trace_part = trace_file
                            else:
                                # For regular trace files with paths
                                trace_part = Path(trace_file).stem
                                # If it's a complex path, get just the last part
                                if "_" in trace_part:
                                    trace_part = trace_part.split("_")[0]
                            if trace_part:
                                parts.append(f"trace_{sanitize_for_filename(trace_part)}")
                        except Exception:
                            # Fallback if path handling fails
                            trace_name = sanitize_for_filename(str(trace_file))
                            parts.append(f"trace_{trace_name[:10]}")  # Limit length

    # If we couldn't generate any meaningful parts, use a random suffix
    if not parts:
        import random

        return f"config_{random.randint(1000, 9999)}"

    return "_".join(parts)


def filter_params_for_engine(section_key, section_data, engine):
    """
    Filters parameters in a section to only include those that are compatible with the specified engine.

    Args:
        section_key: The section key (e.g., 'server', 'model', 'parallel_spec')
        section_data: The section data dictionary
        engine: The engine to filter for

    Returns:
        A new dictionary with only the compatible parameters
    """
    if engine not in ENGINE_PARAM_COMPATIBILITY:
        return section_data  # If engine not found, return all parameters

    compatible_params = ENGINE_PARAM_COMPATIBILITY[engine].get(section_key, [])

    # Always include the engine parameter
    if section_key == "server":
        compatible_params.append("openai_server_engine")

    # Create a new dictionary with only compatible parameters
    filtered_data = {}
    for param, value in section_data.items():
        if param in compatible_params or param == "openai_server_engine":
            filtered_data[param] = value

    return filtered_data


def validate_config_compatibility(config_data):
    """
    Validates that the configuration doesn't contain incompatible combinations.
    For example, if sglang is selected as an engine but pp_dimension > 1 is also selected,
    this would be an incompatible combination since sglang doesn't support pipeline parallelism.

    Args:
        config_data: The configuration data dictionary

    Returns:
        A tuple of (is_valid, warnings) where:
        - is_valid is a boolean indicating if the configuration is valid
        - warnings is a list of warning messages for incompatible combinations
    """
    warnings = []

    # Check if we have multiple engines
    engines = []
    if "server" in config_data and "openai_server_engine" in config_data["server"]:
        engine_value = config_data["server"]["openai_server_engine"]
        if is_multi_value(engine_value):
            engines = get_multi_values(engine_value)
        else:
            engines = [engine_value]

    # Check for incompatible parallel configurations
    if "parallel_spec" in config_data:
        parallel_spec = config_data["parallel_spec"]

        # Check pp_dimension
        if "pp_dimension" in parallel_spec:
            pp_values = get_multi_values(parallel_spec["pp_dimension"])
            has_pp_greater_than_one = any(
                pp > 1 for pp in pp_values if isinstance(pp, (int, float))
            )

            # Check engines that don't support pipeline parallelism
            for engine in engines:
                if (
                    engine in ENGINE_PARAM_COMPATIBILITY
                    and "parallel_spec" in ENGINE_PARAM_COMPATIBILITY[engine]
                ):
                    if (
                        "pp_dimension"
                        not in ENGINE_PARAM_COMPATIBILITY[engine]["parallel_spec"]
                        and has_pp_greater_than_one
                    ):
                        warnings.append(
                            f"Warning: Engine '{engine}' does not support pipeline parallelism (pp_dimension > 1)"
                        )

        # Check tp_dimension
        if "tp_dimension" in parallel_spec:
            tp_values = get_multi_values(parallel_spec["tp_dimension"])
            has_tp_greater_than_one = any(
                tp > 1 for tp in tp_values if isinstance(tp, (int, float))
            )

            # Check engines that don't support tensor parallelism
            for engine in engines:
                if (
                    engine in ENGINE_PARAM_COMPATIBILITY
                    and "parallel_spec" in ENGINE_PARAM_COMPATIBILITY[engine]
                ):
                    if (
                        "tp_dimension"
                        not in ENGINE_PARAM_COMPATIBILITY[engine]["parallel_spec"]
                        and has_tp_greater_than_one
                    ):
                        warnings.append(
                            f"Warning: Engine '{engine}' does not support tensor parallelism (tp_dimension > 1)"
                        )

    # Check for other incompatible parameters
    for section_key in ["server", "model"]:
        if section_key in config_data and isinstance(config_data[section_key], dict):
            section_data = config_data[section_key]

            for param_key, param_value in section_data.items():
                # Skip engine parameter itself
                if param_key == "openai_server_engine":
                    continue

                # Check if this parameter is supported by all engines
                for engine in engines:
                    if (
                        engine in ENGINE_PARAM_COMPATIBILITY
                        and section_key in ENGINE_PARAM_COMPATIBILITY[engine]
                    ):
                        if (
                            param_key
                            not in ENGINE_PARAM_COMPATIBILITY[engine][section_key]
                        ):
                            warnings.append(
                                f"Parameter '{section_key}.{param_key}' is not supported by engine '{engine}'"
                            )

    return len(warnings) == 0, warnings


def sanitize_for_filename(value_str):
    """Sanitizes a string for path/filename use."""
    if not isinstance(value_str, str):
        value_str = str(value_str)
    # Replace spaces and special chars with underscores
    sanitized = re.sub(r"[^\w\-\.]", "_", value_str)
    # Remove leading/trailing underscores and collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "unknown"


def get_experiment_name():
    """Prompts user for a custom experiment name."""
    clear_screen()
    print("--- Experiment saving ---")
    print(
        "Please provide a name (leave blank to use auto-generated name). Setting a custom name will also change the output directory for this set of experiments:"
    )
    name = input("> ").strip()
    return sanitize_for_filename(name) if name else ""


def generate_suggested_names(config_data):
    """Generates suggested dir/filenames based on config."""
    if not config_data or not isinstance(config_data, dict):
        return None, "custom_config.yml"

    server_data = config_data.get("server", {})
    model_data = config_data.get("model", {})
    parallel_data = config_data.get("parallel_spec", {})
    benchmark_data = config_data.get("benchmark_config", {})
    reqgen_data = config_data.get("request_generator_config", {})

    engine = server_data.get("openai_server_engine", "unknown-engine")
    model_name = model_data.get("name", "unknown-model")
    tp = parallel_data.get("tp_dimension", "X")
    pp = parallel_data.get("pp_dimension", "X")
    trace_path_str = reqgen_data.get("trace_request_length_generator_trace_file")
    qps_val = benchmark_data.get("qps")
    if qps_val is None:
        qps_val = reqgen_data.get("start_qps", "X")

    engine_sanitized = sanitize_for_filename(engine)
    model_name_sanitized = sanitize_for_filename(model_name)
    tp_sanitized = sanitize_for_filename(str(tp))
    pp_sanitized = sanitize_for_filename(str(pp))

    derived_trace_name = "no-trace"
    if trace_path_str and isinstance(trace_path_str, str):
        try:
            trace_stem = Path(trace_path_str).stem
            derived_trace_name = (
                trace_stem.split("_")[0] if "_" in trace_stem else trace_stem
            )
            if not derived_trace_name:
                derived_trace_name = "trace"
        except Exception:
            pass
    elif trace_path_str:
        derived_trace_name = str(trace_path_str)
    trace_name_sanitized = sanitize_for_filename(derived_trace_name)

    qps_str = "X"
    if isinstance(qps_val, (int, float)):
        if float(qps_val) == int(qps_val):
            qps_str = str(int(qps_val))
        else:
            qps_str = str(qps_val).replace(".", "_")
    elif qps_val is not None:
        qps_str = sanitize_for_filename(str(qps_val))

    suggested_relative_dir_part = f"{engine_sanitized}_{model_name_sanitized}_tp{tp_sanitized}_pp{pp_sanitized}_{trace_name_sanitized}"
    suggested_filename = f"{engine_sanitized}_{model_name_sanitized}_tp{tp_sanitized}_pp{pp_sanitized}_{trace_name_sanitized}_qps{qps_str}.yml"

    return suggested_relative_dir_part, suggested_filename


def is_multi_value(value):
    """Checks if a value is a multi-value field."""
    return isinstance(value, dict) and MULTI_VALUE_MARKER in value


def get_multi_values(value):
    """Gets the list of values from a multi-value field."""
    if is_multi_value(value):
        return value.get(MULTI_VALUE_MARKER, [])
    return [value] if value is not None else []


def create_multi_value(values):
    """Creates a multi-value field from a list of values."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return {MULTI_VALUE_MARKER: values}


def add_value_to_multi(current_value, new_value):
    """Adds a value to a multi-value field."""
    current_values = get_multi_values(current_value)
    if new_value not in current_values:
        current_values.append(new_value)
    return create_multi_value(current_values)


def remove_value_from_multi(current_value, index):
    """Removes a value from a multi-value field by index."""
    current_values = get_multi_values(current_value)
    if 0 <= index < len(current_values):
        current_values.pop(index)
    return create_multi_value(current_values)


def is_editable(section_key, item_key, options_data):
    """Checks if item is editable based on options config."""
    if not options_data or not isinstance(options_data, dict):
        return False
    section_options = options_data.get(section_key)
    return isinstance(section_options, dict) and item_key in section_options


def get_compatible_engines(section_key, item_key):
    """Returns a list of engines that support the given parameter."""
    compatible_engines = []
    for engine, params in ENGINE_PARAM_COMPATIBILITY.items():
        if section_key in params and item_key in params[section_key]:
            compatible_engines.append(engine)
    return compatible_engines


def display_section_menu(section_name, section_data, options_data):
    """Displays items within a section, marking editable ones."""
    clear_screen()
    print(f"--- Editing Section: {section_name} ---")
    display_items = []
    editable_keys_found = False
    if not isinstance(section_data, dict):
        print("Section data is not a dictionary.")
    else:
        current_display_index = 1
        for key, value in section_data.items():
            if is_multi_value(value):
                values = get_multi_values(value)
                if values:
                    # Format the values as a list with a max length
                    value_strs = [repr(v) for v in values]
                    combined_str = ", ".join(value_strs)
                    max_len = 70
                    if len(combined_str) > max_len:
                        combined_str = combined_str[: max_len - 3] + "..."
                    value_display = f"[{combined_str}]"
                else:
                    value_display = "(empty)"
            elif (
                section_name == "model"
                and key == "chat_template"
                and isinstance(value, str)
                and len(value) > 70
            ):
                value_display = "(long template...)"
            else:
                value_str = repr(value)
                max_len = 70
                display_val = (
                    value_str[: max_len - 3] + "..."
                    if len(value_str) > max_len
                    else value_str
                )
                value_display = display_val
            is_key_editable = is_editable(section_name, key, options_data)
            marker = "*" if is_key_editable else " "
            if is_key_editable:
                editable_keys_found = True

            # Add engine compatibility in parentheses
            engine_info = ""
            if section_name in ["server", "model", "parallel_spec"]:
                compatible_engines = get_compatible_engines(section_name, key)
                if compatible_engines and key != "openai_server_engine":
                    engine_info = f" ({', '.join(compatible_engines)})"

            display = f" {marker}[{current_display_index}] {key}: {value_display}{engine_info}"
            display_items.append((current_display_index, key, display))
            current_display_index += 1
        if not display_items:
            print("  (Section is empty)")
        else:
            for _, key, line in display_items:
                print(line)
    print("\n--- Options ---")
    if editable_keys_found:
        print("  Select number (*) to edit an item.")
    else:
        print("  (No items in this section are configured as editable)")
    print("  [B] Back to Main Menu")
    print("-" * 20)
    return {idx: key for idx, key, _ in display_items}


def edit_section_with_options(config, section_key, options_data, model_mapping_data):
    """UI and logic for editing an item using predefined options."""
    global CONFIG_OPTIONS, MODEL_MAPPING
    if section_key not in config or not isinstance(config.get(section_key), dict):
        show_message(f"Error: Section '{section_key}' not found or is not editable.")
        return
    section_data = config[section_key]
    while True:
        current_options = CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
        current_mapping = MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
        key_map = display_section_menu(section_key, section_data, current_options)
        if not section_data:
            show_message("Section is currently empty.")

        # Define allowed characters based on the number of items
        allowed_chars = ["b"] + [str(i) for i in key_map.keys() if i < 10]

        choice = get_single_key(
            "Enter choice (number to edit, or 'B'): ", allowed_chars
        )
        if choice == "b":
            break
        elif choice.isdigit():
            try:
                item_index = int(choice)
                item_key = key_map.get(item_index)
                if item_key:
                    if not is_editable(section_key, item_key, current_options):
                        show_message(
                            f"Item '{item_key}' is not configured as editable."
                        )
                        continue
                    available_options = current_options.get(section_key, {}).get(
                        item_key, []
                    )
                    if not isinstance(available_options, list):
                        show_message(
                            f"Warning: Options for '{section_key}.{item_key}' not a list."
                        )
                        continue
                    current_value = section_data.get(item_key)

                    # Handle multi-value editing
                    edit_multi_value_field(
                        section_key,
                        item_key,
                        section_data,
                        current_value,
                        available_options,
                        current_mapping,
                    )
                    continue
                else:
                    show_message(f"Invalid selection number '{choice}'.")
                    continue
            except ValueError:
                show_message("Invalid input.")
                continue
            except Exception as e:
                show_message(f"Editing error: {e}")
                continue
        else:
            show_message("Invalid choice.")
            continue


def edit_multi_value_field(
    section_key,
    item_key,
    section_data,
    current_value,
    available_options,
    current_mapping,
):
    """UI for editing a multi-value field."""
    while True:
        clear_screen()
        print(f"--- Editing: {section_key}.{item_key} ---")

        # Display engine compatibility for this parameter in parentheses
        if section_key in ["server", "model", "parallel_spec"]:
            compatible_engines = get_compatible_engines(section_key, item_key)
            if compatible_engines and item_key != "openai_server_engine":
                print(f"Parameter compatibility: ({', '.join(compatible_engines)})")

        # Display current values - always get the latest values from section_data
        current_value = section_data.get(item_key)
        current_values = get_multi_values(current_value)
        print("\nCurrent Values:")
        if not current_values:
            print("  (No values set)")
        else:
            for i, val in enumerate(current_values):
                value_str = repr(val)
                max_len = 70
                display_val = (
                    value_str[: max_len - 3] + "..."
                    if len(value_str) > max_len
                    else value_str
                )
                print(f"  [{i+1}] {display_val}")

        # Display available options
        print("\nAvailable Options:")
        if not available_options:
            print("  (No predefined options found)")
        else:
            for i, option in enumerate(available_options):
                print(f"  [{i+1}] {repr(option)}")

        # Display actions
        print("\n--- Actions ---")
        print("  [A] Add a value")
        if current_values:
            print("  [R] Remove a value")
        print("  [B] Back")
        print("-" * 20)

        # Get user choice
        allowed_chars = ["a", "b"] + (["r"] if current_values else [])

        choice = get_single_key("Select action: ", allowed_chars)

        if choice == "b":
            break
        elif choice == "a":
            # Add a new value
            if not available_options:
                show_message("No options available to add.")
                continue

            # Display options for adding
            clear_screen()
            print(f"--- Add Value to {section_key}.{item_key} ---")
            print("Select an option to add:")
            for i, option in enumerate(available_options):
                print(f"  [{i+1}] {repr(option)}")

            # Add option to type a custom value
            print("  [T] Type custom value")
            print("  [B] Back")
            print("-" * 20)

            # Define allowed characters for option selection
            option_allowed_chars = ["b", "t"] + [
                str(i + 1) for i in range(min(9, len(available_options)))
            ]

            option_choice = get_single_key(
                "Select option number, [T]ype custom, or [B]ack: ", option_allowed_chars
            )
            if option_choice == "b":
                continue

            if option_choice == "t":
                # Allow user to type a custom value
                clear_screen()
                print(f"--- Type Custom Value for {section_key}.{item_key} ---")
                print("Enter a custom value (or leave blank to cancel):")
                custom_input = input("> ").strip()

                if not custom_input:
                    continue

                # Try to convert the input to an appropriate type
                try:
                    # First check if it's a boolean
                    if custom_input.lower() in ["true", "false"]:
                        new_value = custom_input.lower() == "true"
                    # Then check if it's an integer
                    elif custom_input.isdigit() or (
                        custom_input.startswith("-") and custom_input[1:].isdigit()
                    ):
                        new_value = int(custom_input)
                    # Then check if it's a float
                    elif "." in custom_input:
                        try:
                            new_value = float(custom_input)
                        except ValueError:
                            new_value = custom_input
                    else:
                        new_value = custom_input

                    updated_value = add_value_to_multi(current_value, new_value)
                    section_data[item_key] = updated_value

                    # Special handling for model name
                    if (
                        section_key == "model"
                        and item_key == "name"
                        and not is_multi_value(updated_value)
                    ):
                        apply_model_mapping(section_data, new_value, current_mapping)

                    show_message(f"Added custom value: {repr(new_value)}")
                except ValueError as e:
                    show_message(f"Invalid input: {e}")
            elif option_choice.isdigit():
                selected_index = int(option_choice) - 1
                if 0 <= selected_index < len(available_options):
                    new_value = available_options[selected_index]
                    updated_value = add_value_to_multi(current_value, new_value)
                    section_data[item_key] = updated_value

                    # Special handling for model name
                    if (
                        section_key == "model"
                        and item_key == "name"
                        and not is_multi_value(updated_value)
                    ):
                        apply_model_mapping(section_data, new_value, current_mapping)

                    show_message(f"Added value: {repr(new_value)}")
                else:
                    show_message("Invalid index.")
            else:
                show_message("Invalid choice.")

        elif choice == "r" and current_values:
            # Remove a value
            clear_screen()
            print(f"--- Remove Value from {section_key}.{item_key} ---")
            print("Select a value to remove:")
            for i, val in enumerate(current_values):
                value_str = repr(val)
                max_len = 70
                display_val = (
                    value_str[: max_len - 3] + "..."
                    if len(value_str) > max_len
                    else value_str
                )
                print(f"  [{i+1}] {display_val}")
            print("  [B] Back")
            print("-" * 20)

            # Define allowed characters for removal selection
            remove_allowed_chars = ["b"] + [
                str(i + 1) for i in range(min(9, len(current_values)))
            ]

            remove_choice = get_single_key(
                "Select value to remove or [B]ack: ", remove_allowed_chars
            )
            if remove_choice == "b":
                continue

            if remove_choice.isdigit():
                remove_index = int(remove_choice) - 1
                if 0 <= remove_index < len(current_values):
                    removed_value = current_values[remove_index]
                    updated_value = remove_value_from_multi(current_value, remove_index)
                    section_data[item_key] = updated_value
                    show_message(f"Removed value: {repr(removed_value)}")
                else:
                    show_message("Invalid index.")
            else:
                show_message("Invalid choice.")


def apply_model_mapping(section_data, model_name, current_mapping):
    """Apply model mapping for a specific model name."""
    show_message("Applying model mapping...")
    model_details = current_mapping.get(model_name)
    if model_details and isinstance(model_details, dict):
        id_changed, tpl_changed = False, False
        new_id = model_details.get("identifier")
        new_tpl = model_details.get("chat_template")
        if new_id is not None and section_data.get("identifier") != new_id:
            section_data["identifier"] = new_id
            id_changed = True
            show_message(f"  > Updated 'identifier' to: {repr(new_id)}")
        if new_tpl is not None and section_data.get("chat_template") != new_tpl:
            section_data["chat_template"] = new_tpl
            tpl_changed = True
            show_message(f"  > Updated 'chat_template' (in memory).")
        if not id_changed and not tpl_changed:
            show_message("  > No identifier or template changes needed from mapping.")
    else:
        show_message(f"  > Warning: No mapping details found for model '{model_name}'.")


# --- Display and Menu Functions ---


def display_main_menu(config):
    """Displays the main menu."""
    clear_screen()
    print("--- Config editor ---")
    print("Current configuration:")
    top_keys = []
    if not config:
        print("\n  (No configuration loaded)")
    else:
        top_keys = list(config.keys())
        for i, key in enumerate(top_keys):
            print(f"\n[{i+1}] {key}:")
            section_data = config.get(key)
            if isinstance(section_data, dict):
                if not section_data:
                    print("    (empty)")
                else:
                    for sub_key, sub_value in section_data.items():
                        if is_multi_value(sub_value):
                            values = get_multi_values(sub_value)
                            if values:
                                # Format the values as a list with a max length
                                value_strs = [repr(v) for v in values]
                                combined_str = ", ".join(value_strs)
                                max_len = 65
                                if len(combined_str) > max_len:
                                    combined_str = combined_str[: max_len - 3] + "..."
                                print(f"    {sub_key}: [{combined_str}]")
                            else:
                                print(f"    {sub_key}: (empty)")
                        elif (
                            key == "model"
                            and sub_key == "chat_template"
                            and isinstance(sub_value, str)
                            and len(sub_value) > 65
                        ):
                            print(f"    {sub_key}: (long template...)")
                        else:
                            value_str = repr(sub_value)
                            max_len = 65
                            display_val = (
                                value_str[: max_len - 3] + "..."
                                if len(value_str) > max_len
                                else value_str
                            )
                            print(f"    {sub_key}: {display_val}")
            elif isinstance(section_data, list):
                print(f"    (List with {len(section_data)} items)")
            elif section_data is None:
                print("    (None)")
            else:
                value_str = repr(section_data)
                max_len = 70
                display_val = (
                    value_str[: max_len - 3] + "..."
                    if len(value_str) > max_len
                    else value_str
                )
                print(f"    Value: {display_val}")
    print("\n\n--- Options ---")
    print("  [1..N] Edit Section | [S] Save | [R] Reset | [Q] Quit")
    print("-" * 20)
    return top_keys


def get_single_key(prompt="", allowed_chars=None):
    """Gets a single keypress from the user without requiring Enter.

    Args:
        prompt: Text to display before getting input
        allowed_chars: List of allowed characters, or None to allow any

    Returns:
        The character pressed by the user
    """
    if prompt:
        print(prompt, end="", flush=True)

    while True:
        char = getch().lower()
        if allowed_chars is None or char in allowed_chars:
            print(char)  # Echo the character
            return char


# --- Main execution loop ---
def main():
    global CONFIG_OPTIONS, MODEL_MAPPING
    print("Loading initial configurations...")
    CONFIG_OPTIONS = load_yaml_file(OPTIONS_CONFIG_PATH, "options configuration")
    if CONFIG_OPTIONS is None:
        sys.exit(f"Fatal Error: {OPTIONS_CONFIG_PATH}")
    if not CONFIG_OPTIONS:
        print(f"Warning: Options file '{OPTIONS_CONFIG_PATH}' empty/invalid.")
    MODEL_MAPPING = load_yaml_file(MODEL_MAPPING_PATH, "model mapping")
    if MODEL_MAPPING is None:
        sys.exit(f"Fatal Error: {MODEL_MAPPING_PATH}")
    if not MODEL_MAPPING:
        print(f"Warning: Mapping file '{MODEL_MAPPING_PATH}' empty/invalid.")
    current_config = load_yaml_file(DEFAULT_CONFIG_PATH, "default configuration")
    if current_config is None:
        sys.exit(f"Fatal Error: {DEFAULT_CONFIG_PATH}")
    if not current_config:
        print(f"Warning: Default config '{DEFAULT_CONFIG_PATH}' empty/invalid.")
    show_message("\nInitialization complete.")

    while True:
        options = CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
        mapping = MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
        top_keys = display_main_menu(current_config)

        # Define allowed characters for main menu
        allowed_chars = ["q", "s", "r"] + [
            str(i + 1) for i in range(min(9, len(top_keys)))
        ]

        choice = get_single_key("Enter choice: ", allowed_chars)

        if choice == "q":
            break
        elif choice == "s":
            if not current_config:
                show_message("Cannot save empty config.")
                continue

            # Get custom experiment name from user
            custom_name = get_experiment_name()

            suggested_rel_dir, suggested_filename = generate_suggested_names(
                current_config
            )
            if suggested_rel_dir is None:
                show_message("Warning: Could not generate suggested names.")
                suggested_rel_dir = "unknown_dir"
            if "benchmark_config" in current_config and isinstance(
                current_config["benchmark_config"], dict
            ):
                full_output_dir_str = (
                    Path(BENCHMARK_BASE_DIR) / suggested_rel_dir
                ).as_posix()
                if (
                    current_config["benchmark_config"].get("output_dir")
                    != full_output_dir_str
                ):
                    show_message(
                        f"\nAuto-updating 'benchmark_config.output_dir': {full_output_dir_str}"
                    )
                    current_config["benchmark_config"][
                        "output_dir"
                    ] = full_output_dir_str
                    current_config["benchmark_config"]["should_use_given_dir"] = True
                else:
                    show_message(
                        f"\n'benchmark_config.output_dir' set to: {full_output_dir_str}"
                    )
                    current_config["benchmark_config"]["should_use_given_dir"] = True
            else:
                show_message(
                    "\nWarning: 'benchmark_config' missing/invalid; cannot update output_dir."
                )

            # Modify filename with custom name if provided
            if custom_name:
                # Extract extension from suggested filename
                name_parts = suggested_filename.rsplit(".", 1)
                base_name = name_parts[0]
                extension = name_parts[1] if len(name_parts) > 1 else "yml"
                final_filename = f"{custom_name}_{base_name}.{extension}"
                show_message(f"Using custom experiment name: {custom_name}")
            else:
                final_filename = suggested_filename
                show_message(f"Using suggested config filename: {final_filename}")

            if final_filename:
                success, saved_filepaths = save_config(
                    current_config, final_filename, experiment_name=custom_name
                )
            else:
                show_message("Save cancelled (error generating filename).")

        elif choice == "r":
            confirm = get_single_key(
                f"Reset config using '{DEFAULT_CONFIG_PATH}'? (y/N): ", ["y", "n"]
            )
            if confirm == "y":
                show_message(f"Reloading default...")
                loaded_default = load_yaml_file(DEFAULT_CONFIG_PATH, "default")
                if loaded_default is not None:
                    if isinstance(loaded_default, dict):
                        current_config = loaded_default
                        show_message("Reset ok.")
                    else:
                        show_message(f"Error: Default file invalid structure.")
            else:
                show_message("Reset cancelled.")
        elif choice.isdigit():
            try:
                section_index = int(choice) - 1
                if 0 <= section_index < len(top_keys):
                    selected_key = top_keys[section_index]
                    current_options_state = (
                        CONFIG_OPTIONS if isinstance(CONFIG_OPTIONS, dict) else {}
                    )
                    current_mapping_state = (
                        MODEL_MAPPING if isinstance(MODEL_MAPPING, dict) else {}
                    )
                    edit_section_with_options(
                        current_config,
                        selected_key,
                        current_options_state,
                        current_mapping_state,
                    )
                else:
                    show_message("Invalid section number.")
            except ValueError:
                show_message("Invalid input.")
            except Exception as e:
                show_message(f"Error: {e}")
        else:
            show_message("Invalid choice.")

    print("\nExiting config editor.")


def config_editor_entrypoint():
    """Entry point for the config editor when run as a module."""
    try:
        # Check if experiment cache is available
        if CACHE_AVAILABLE:
            cache = load_experiment_cache()
            cache_count = len(cache.get("completed_experiments", []))
            print(f"Experiment cache loaded: {cache_count} completed experiments")

        main()
    except KeyboardInterrupt:
        print("\nConfig editor exited.")
    except Exception as e:
        print(f"Error in config editor: {e}")
        import traceback

        traceback.print_exc()
