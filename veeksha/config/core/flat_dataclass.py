import json
import sys
import copy
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from itertools import product
from typing import Any, Dict, List, Optional, get_args

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.decorators import has_allow_from_file_decorator
from veeksha.config.utils import (
    get_all_subclasses,
    get_inner_type,
    is_bool,
    is_composed_of_primitives,
    is_dict,
    is_list,
    is_optional,
    is_primitive_type,
    is_subclass,
    load_yaml_config,
    to_snake_case,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


def explode_dict(config: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
    """
    Recursively explode a dictionary containing lists of values into a list of dictionaries
    representing all combinations (cartesian product), with optional prefix applied to keys.
    
    Args:
        config: Dictionary potentially containing lists to explode
        prefix: Prefix to apply to all top-level keys
    
    Example:
        Input: {'a': [1, 2], 'b': [3, 4]}, prefix='test_'
        Output: [{'test_a': 1, 'test_b': 3}, {'test_a': 1, 'test_b': 4}, 
                 {'test_a': 2, 'test_b': 3}, {'test_a': 2, 'test_b': 4}]
    """
    def _explode_dict_recursive(d: Dict[str, Any], level: int = 0) -> List[Dict[str, Any]]:
        list_keys = []
        list_values = []
        non_list_items = {}
        dict_items = {}
        
        for key, value in d.items():
            if isinstance(value, list) and len(value) > 0:
                # Check if it's a list of configs (dicts) or primitive values
                if isinstance(value[0], dict):
                    # List of configs - need to recursively explode each one
                    exploded_configs = []
                    for config in value:
                        exploded = _explode_dict_recursive(config, level + 1)
                        exploded_configs.extend(exploded)
                    list_keys.append(key)
                    list_values.append(exploded_configs)
                else:
                    # List of primitive values - create combinations
                    list_keys.append(key)
                    list_values.append(value)
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                dict_items[key] = value
            else:
                non_list_items[key] = value
        
        # First, handle dict items recursively
        dict_combinations = [{}]
        
        for key, nested_dict in dict_items.items():
            exploded_nested = _explode_dict_recursive(nested_dict, level + 1)
            new_combinations = []
            for base_combo in dict_combinations:
                for nested_combo in exploded_nested:
                    new_combo = base_combo.copy()
                    new_combo[key] = nested_combo
                    new_combinations.append(new_combo)
            dict_combinations = new_combinations
        
        # If no lists found at this level, combine with dict combinations
        if not list_keys:
            result = []
            for dict_combo in dict_combinations:
                combined = non_list_items.copy()
                combined.update(dict_combo)
                result.append(combined)
            return result
        
        # Generate all combinations including lists
        result = []
        for combination in product(*list_values):
            for dict_combo in dict_combinations:
                new_config = non_list_items.copy()
                new_config.update(dict_combo)
                for key, value in zip(list_keys, combination):
                    new_config[key] = value
                result.append(new_config)
        
        return result
    
    def add_prefix_to_dict(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to all keys in a dictionary """
        def _add_prefix_recursive(data: Dict[str, Any], current_prefix: str = "") -> Dict[str, Any]:
            result = {}
            
            for key, value in data.items():
                # TODO: this is a hack to handle the case where the config has a 'type' key
                # for robustness, we should fetch the actual name of the class of this particular type
                if "type" in data and key != "type":
                    # i.e. request_generator_config -> trace_request_generator_config
                    prefixed_key = f"{data['type']}_{current_prefix}{key}"
                else:
                    prefixed_key = f"{current_prefix}{key}"
                
                if isinstance(value, dict):
                    # For nested dicts, recursively process with composed prefix
                    flattened = _add_prefix_recursive(value, f"{prefixed_key}_")
                    result.update(flattened)
                else:
                    # Leaf value - add it with the full prefix
                    result[prefixed_key] = value
                    
            return result
            
        return _add_prefix_recursive(d, prefix)
    
    # Handle special case where config has a '_list' key (from load_yaml_config)
    if isinstance(config, dict) and len(config) == 1 and '_list' in config:
        list_data = config['_list']
        if isinstance(list_data, list):
            # Treat each item in the list as a separate config
            all_exploded = []
            for item in list_data:
                if isinstance(item, dict):
                    exploded = _explode_dict_recursive(item)
                    all_exploded.extend(exploded)
                else:
                    # Non-dict items are wrapped
                    all_exploded.append({'_value': item})
            return [add_prefix_to_dict(cfg, prefix) for cfg in all_exploded]
    
    return [add_prefix_to_dict(config, prefix) for config in _explode_dict_recursive(config)]


def topological_sort(dataclass_dependencies: dict) -> list:
    in_degree = defaultdict(int)
    for cls, dependencies in dataclass_dependencies.items():
        for dep in dependencies:
            in_degree[dep] += 1

    zero_in_degree_classes = deque(
        [cls for cls in dataclass_dependencies if in_degree[cls] == 0]
    )
    sorted_classes = []

    while zero_in_degree_classes:
        cls = zero_in_degree_classes.popleft()
        sorted_classes.append(cls)
        for dep in dataclass_dependencies[cls]:
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                zero_in_degree_classes.append(dep)

    return sorted_classes


def overwrite_args_with_config(
    args: dict,
    config: dict,
    keys_to_file_field_names: Dict[str, str],
    default_values: dict,
    cli_provided_args: set
):
    """Overwrite args with values from file_config in a DFS manner"""

    for key, value in config.items():
        if isinstance(value, dict):
            # a nested config object: we compose the prefix
            overwrite_args_with_config(
                args,
                value,
                keys_to_file_field_names,
                default_values,
                cli_provided_args
            )
            continue

        # Ignore keys that are not recognised by the FlatClass.
        if not hasattr(args, key):
            logger.warning(
                f"Arg {key} provided by {keys_to_file_field_names[key]} not found in supported args."
            )
            continue

        if key not in cli_provided_args:
            # TODO: verbosity level for this (f"Overwriting {key} with {value}")
            setattr(args, key, value)
        else:
            logger.warning(
                f"Arg {key} provided by {keys_to_file_field_names[key]} set via CLI. Skipped overwrite."
            )


def reconstruct_original_dataclass(self) -> Any:
    """
    This function is dynamically mapped to FlatClass as an instance method.
    Reconstructs the original dataclass from the flattened representation.
    """
    # list of classes, from the most dependent to the least dependent
    sorted_classes = topological_sort(self.dataclass_dependencies)

    instances = {}

    # iter over classes from least dependent to most
    for _cls in reversed(sorted_classes):
        args = {}
        # instantiate current class fields
        for prefixed_field_name, original_field_name, field_type in self.dataclass_args[
            _cls
        ]:
            if is_subclass(field_type, BasePolyConfig):
                config_type = getattr(self, f"{prefixed_field_name}_type")
                # find all subclasses of field_type and check which one matches the config_type
                config_type_matched = False
                # base poly children cointains all subclasses of the base poly config
                for child_name, child_cls in self.base_poly_children[
                    prefixed_field_name
                ].items():
                    if str(child_cls.get_type()) == config_type:
                        config_type_matched = True
                        args[original_field_name] = instances[child_name]
                        break
                assert (
                    config_type_matched
                ), f"Invalid type {config_type} for {prefixed_field_name}_type. Valid types: {[str(subclass.get_type()) for subclass in get_all_subclasses(field_type)]}"
            # child dataclass has already been instantiated, so just assign it
            elif hasattr(field_type, "__dataclass_fields__"):
                args[original_field_name] = instances[prefixed_field_name]
            # primitive type
            else:
                value = getattr(self, prefixed_field_name)
                if value is not MISSING and callable(value):
                    # to handle default factory values
                    value = value()
                args[original_field_name] = value

        instances[_cls] = self.dataclass_names_to_classes[_cls](**args)

    return instances[sorted_classes[0]]


@classmethod
def create_from_cli_args(cls) -> Any:
    """
    This function is dynamically mapped to FlatClass as a class method.
    
    Returns:
        List[cls]: A list of instances of the dataclass, one for each combination of configs created.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    all_default_values = {}
    argnames_to_field_names = {}

    for field in fields(cls):
        nargs = None
        action = None
        field_type = field.type
        help_text = cls.metadata_mapping[field.name].get("help", None)
        argname = cls.metadata_mapping[field.name].get("argname", None)
        if argname in argnames_to_field_names:
            raise ValueError(
                f"Cannot have multiple fields with the same argname: {argname} already exists for field {argnames_to_field_names[argname]}"
            )
        elif argname is not None:
            argnames_to_field_names[argname] = field.name

        is_field_optional = is_optional(field.type)

        if is_field_optional:
            field_type = get_inner_type(field.type)

        if is_list(field_type):
            assert is_composed_of_primitives(field_type)
            field_type = get_args(field_type)[0]
            if is_primitive_type(field_type):
                nargs = "+"
            else:
                field_type = json.loads
        elif is_dict(field_type):
            assert is_composed_of_primitives(field_type)
            field_type = json.loads
        elif is_bool(field_type):
            action = BooleanOptionalAction

        arg_params = {
            "type": field_type,
            "action": action,
            "help": help_text,
        }

        # handle cases with default and default factory args
        if field.default is not MISSING:
            value = field.default
            if callable(value):
                value = value()
            arg_params["default"] = value
            all_default_values[field.name] = value
        elif field.default_factory is not MISSING:
            arg_params["default"] = field.default_factory()
            all_default_values[field.name] = field.default_factory()
        else:
            all_default_values[field.name] = object()  # sentinel value
            if is_field_optional:
                arg_params["default"] = None
            else:
                arg_params["required"] = True

        if nargs:
            arg_params["nargs"] = nargs
        cli_arg_name = field.name.replace("_", "-") if argname is None else argname
        parser.add_argument(f"--{cli_arg_name}", dest=field.name, **arg_params)

    args = parser.parse_args()
    cli_provided_args = set()
    # get args explicitly provided by CLI
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--"):
            # remove --
            arg_name = arg[2:].split("=")[0]
            # check if this maps to a field name via argname mapping
            if arg_name in argnames_to_field_names:
                cli_provided_args.add(argnames_to_field_names[arg_name])
            else:
                # convert - to _
                field_name = arg_name.replace("-", "_")
                if hasattr(args, field_name):
                    cli_provided_args.add(field_name)

    loaded_configs: Dict[str, List[Dict[str, Any]]] = {}  # maps file arg name to list of configs

    logger.info("--------------------------------")
    logger.info("BEGIN LOADING ARGS FROM FILES")
    logger.info("--------------------------------")
    # load config files and overwrite fields not set via CLI
    for file_field_name in cls.dataclass_file_fields.values():
        file_path = getattr(args, file_field_name, None)
        if not file_path:
            continue
        file_config = load_yaml_config(file_path)
        # we want relative file arg names to be mapped to the absolute field names
        # CLI args are provided without the root class prefix except for the root _from_file arg,
        name_of_class_for_file = file_field_name.replace("_from_file", "").replace(
            "-", ""
        )
        if name_of_class_for_file == cls.root_dataclass_name:
            prefix = ""
        else:
            prefix = f"{name_of_class_for_file}_"
            
        loaded_configs[file_field_name] = explode_dict(file_config, prefix)
        
    total_configs = 0
    for file_field_name, configs in loaded_configs.items():
        n_configs = len(configs)
        logger.info(f"File field name: {file_field_name}. Expanded to {n_configs} configs.")
        total_configs += n_configs
    
    # cartesian product of all configs in loaded_configs. 
    # loaded config dicts are already flattened, so we just need to combine them
    # i.e. {a: [1, 2], b: [3, 4]} -> [{1, 3}, {1, 4}, {2, 3}, {2, 4}] (numbers represent flat dicts)
    all_config_combinations = []
    all_keys_to_file_field_names: List[Dict[str, str]] = []
    
    if loaded_configs:
        # Get config lists for cartesian product
        config_lists, file_field_names = list(loaded_configs.values()), list(loaded_configs.keys())
        
        # Generate all combinations using itertools.product
        for combination in product(*config_lists):
            # Combine all configs in this combination into a single dict
            combined_config = {}
            params_to_files = {}
            for i, config in enumerate(combination):
                current_file_field_name = file_field_names[i]
                # Check for conflicts between configs
                for key, value in config.items():
                    if key in combined_config:
                        raise ValueError(f"Arg {key} provided by {current_file_field_name} is also set by {params_to_files[key]}.")
                    combined_config[key] = value
                    params_to_files[key] = current_file_field_name
            all_config_combinations.append(combined_config)
            all_keys_to_file_field_names.append(params_to_files)
        
        logger.info(f"Created {len(all_config_combinations)} total config combinations")
        logger.info("--------------------------------")
        logger.info("END LOADING ARGS FROM FILES")
        logger.info("--------------------------------")
        
        # Now, we overwrite args with all combinations, returning a list of full configs
        final_args = []
        for i, config in enumerate(all_config_combinations):
            keys_to_file_field_names = all_keys_to_file_field_names[i]
            args_copy = copy.deepcopy(args)
            overwrite_args_with_config(
                args_copy,
                config,
                keys_to_file_field_names,
                all_default_values,
                cli_provided_args
            )
            final_args.append(args_copy)
    else:
        final_args = [args]

    return [cls(**vars(arg_instance)) for arg_instance in final_args]


def get_config_class_by_type_name(config_class: Any, type_name: str) -> Any:
    for subclass in get_all_subclasses(config_class):
        if subclass.get_type().name.upper() == type_name.upper():
            return subclass

    raise ValueError(f"Config class with name {type_name} not found.")


def create_flat_dataclass(input_dataclass: Any) -> Any:
    """
    Creates a new FlatClass type by recursively flattening the input dataclass.
    This allows for easy parsing of command line arguments along with storing/loading the configuration to/from a file.
    """
    meta_fields_with_defaults = []
    meta_fields_without_defaults = []
    dataclass_args = defaultdict(list)
    dataclass_dependencies = defaultdict(
        list
    )  # maps unique nested dataclass names to their uniquely named dependencies
    metadata_mapping = {}
    dataclass_file_fields = {}  # maps dataclass to its file field name
    dataclass_names_to_classes = (
        {}
    )  # maps unique nested dataclass names to their corresponding dataclass class
    base_poly_children: Dict[str, Dict[str, Any]] = (
        {}
    )  # maps unique (by name) base poly configs to their children

    def add_file_arg(target_cls, file_field_name: str):
        # adding the implicit "<prefix>_from_file" CLI flag.
        meta_fields_with_defaults.append((file_field_name, Optional[str], None))
        metadata_mapping[file_field_name] = {
            "help": f"Path to YAML/JSON configuration file for {target_cls.__name__}."
        }
        dataclass_file_fields[target_cls] = file_field_name

    def process_dataclass(_input_dataclass, prefix=""):
        """Creates a flattened representation of the input dataclass, adding _from_file and _type fields in some cases and populating metadata fields"""
        prefixed_input_dataclass = (
            f"{prefix[:-1]}" if prefix else f"{to_snake_case(input_dataclass.__name__)}"
        )
        _ = dataclass_dependencies[prefixed_input_dataclass]
        dataclass_names_to_classes[prefixed_input_dataclass] = _input_dataclass

        # add _from_file to non-poly or children of poly dataclasses with explicit @allow_from_file
        if has_allow_from_file_decorator(_input_dataclass):
            file_field_name = (
                f"{prefix}from_file"
                if prefix
                else f"{to_snake_case(_input_dataclass.__name__)}_from_file"
            )
            add_file_arg(_input_dataclass, file_field_name)

        for field in fields(_input_dataclass):
            prefixed_name = f"{prefix}{field.name}"

            if is_optional(field.type):  # type: ignore
                field_type = get_inner_type(field.type)  # type: ignore
            else:
                field_type = field.type

            # if field is a BasePolyConfig, add a type argument and process its children
            if is_subclass(field_type, BasePolyConfig):
                dataclass_args[prefixed_input_dataclass].append(
                    (prefixed_name, field.name, field_type)
                )
                base_poly_children[prefixed_name] = {}

                type_field_name = f"{prefixed_name}_type"

                if field.default_factory is not MISSING:
                    default_value = str(field.default_factory().get_type())  # type: ignore
                elif field.default is not MISSING:
                    if field.default is None:
                        default_value = "None"
                    else:
                        default_value = str(field.default.get_type())  # type: ignore
                else:
                    raise ValueError(
                        f"Field {field.name} of type {field_type} must have a default or default_factory"
                    )

                meta_fields_with_defaults.append(
                    (type_field_name, type(default_value), default_value)
                )
                metadata_mapping[type_field_name] = field.metadata

                # add _from_file to base poly config with explicit @allow_from_file
                if has_allow_from_file_decorator(field_type):
                    file_field_name = f"{prefixed_name}_from_file"
                    add_file_arg(field_type, file_field_name)

                assert hasattr(field_type, "__dataclass_fields__")
                for subclass in get_all_subclasses(field_type):
                    base_poly_children[prefixed_name][
                        prefix + to_snake_case(subclass.__name__)
                    ] = subclass
                    dataclass_dependencies[prefixed_input_dataclass].append(
                        prefix + to_snake_case(subclass.__name__)
                    )
                    process_dataclass(
                        subclass, f"{to_snake_case(prefix[:-1] + subclass.__name__)}_"
                    )
                continue
            # if field is a dataclass, recursively process it
            if hasattr(field_type, "__dataclass_fields__"):
                dataclass_dependencies[prefixed_input_dataclass].append(
                    prefix + to_snake_case(field_type.__name__)
                )
                dataclass_args[prefixed_input_dataclass].append(
                    (prefixed_name, field.name, field_type)
                )
                process_dataclass(field_type, f"{prefix + to_snake_case(field_type.__name__)}_")  # type: ignore
                continue

            field_default = field.default if field.default is not MISSING else MISSING
            field_default_factory = (
                field.default_factory
                if field.default_factory is not MISSING
                else MISSING
            )

            if field_default is not MISSING:
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default)
                )
            elif field_default_factory is not MISSING:
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default_factory)
                )
            else:
                meta_fields_without_defaults.append((prefixed_name, field_type))

            dataclass_args[prefixed_input_dataclass].append(
                (prefixed_name, field.name, field_type)
            )
            metadata_mapping[prefixed_name] = field.metadata

    process_dataclass(input_dataclass)

    meta_fields = meta_fields_without_defaults + meta_fields_with_defaults

    # Flat dataclass with all default values (unitialized)
    FlatClass = make_dataclass("FlatClass", meta_fields)

    # Metadata fields
    FlatClass.dataclass_args = dataclass_args
    FlatClass.dataclass_dependencies = dataclass_dependencies
    FlatClass.dataclass_names_to_classes = dataclass_names_to_classes
    FlatClass.metadata_mapping = metadata_mapping
    FlatClass.dataclass_file_fields = dataclass_file_fields
    FlatClass.base_poly_children = base_poly_children
    FlatClass.root_dataclass_name = to_snake_case(input_dataclass.__name__)
    # Helper methods
    FlatClass.reconstruct_original_dataclass = reconstruct_original_dataclass
    FlatClass.create_from_cli_args = create_from_cli_args

    return FlatClass
