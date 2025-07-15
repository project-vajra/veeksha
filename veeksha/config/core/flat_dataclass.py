import copy
import json
import sys
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, get_args

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


def explode_dict(cls, config: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
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

    def _categorize_dict_items(d: Dict[str, Any]) -> tuple:
        """Categorize dictionary items into lists, dicts, and primitives."""
        list_keys = []
        list_values = []
        non_list_items = {}
        dict_items = {}

        for key, value in d.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict):
                    # list of config dictionaries
                    list_keys.append(key)
                    list_values.append(value)
                else:
                    # list of primitive values
                    list_keys.append(key)
                    list_values.append(value)
            elif isinstance(value, dict):
                dict_items[key] = value
            else:
                non_list_items[key] = value

        return list_keys, list_values, non_list_items, dict_items

    def _explode_dict_list(
        dict_list: List[Dict[str, Any]], level: int
    ) -> List[Dict[str, Any]]:
        """Explode a list of dictionaries recursively."""
        exploded_configs = []
        for config in dict_list:
            exploded = _explode_dict_recursive(config, level + 1)
            exploded_configs.extend(exploded)
        return exploded_configs

    def _generate_dict_combinations(
        dict_items: Dict[str, Dict[str, Any]], level: int
    ) -> List[Dict[str, Any]]:
        """Generate all combinations from nested dictionaries."""
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

        return dict_combinations

    def _combine_non_list_items(
        non_list_items: Dict[str, Any], dict_combinations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine non-list items with dictionary combinations."""
        result = []
        for dict_combo in dict_combinations:
            combined = non_list_items.copy()
            combined.update(dict_combo)
            result.append(combined)
        return result

    def _generate_all_combinations(
        list_keys: List[str],
        list_values: List[List[Any]],
        non_list_items: Dict[str, Any],
        dict_combinations: List[Dict[str, Any]],
        level: int,
    ) -> List[Dict[str, Any]]:
        """Generate all combinations including list values."""
        # handle list of config dictionaries vs primitives
        processed_list_values = []
        for values in list_values:
            if values and isinstance(values[0], dict):
                # explode each config dict in the list
                processed_list_values.append(_explode_dict_list(values, level))
            else:
                # keep primitive values as-is
                processed_list_values.append(values)

        result = []
        for combination in product(*processed_list_values):
            for dict_combo in dict_combinations:
                new_config = non_list_items.copy()
                new_config.update(dict_combo)

                for key, value in zip(list_keys, combination):
                    new_config[key] = value

                result.append(new_config)

        return result

    def _explode_dict_recursive(
        d: Dict[str, Any], level: int = 0
    ) -> List[Dict[str, Any]]:
        """Recursively explode a dictionary into all combinations."""
        list_keys, list_values, non_list_items, dict_items = _categorize_dict_items(d)

        # generate combinations from nested dictionaries
        dict_combinations = _generate_dict_combinations(dict_items, level)

        # if no lists found, just combine non-list items with dict combinations
        if not list_keys:
            return _combine_non_list_items(non_list_items, dict_combinations)

        # generate all combinations including lists
        return _generate_all_combinations(
            list_keys, list_values, non_list_items, dict_combinations, level
        )

    def _add_prefix_to_dict(cls, d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Add prefix to all keys in a dictionary."""

        def _add_prefix_recursive(
            cls, data: Dict[str, Any], current_prefix: str = ""
        ) -> Dict[str, Any]:
            result = {}

            for key, value in data.items():
                if "type" in data and key != "type":
                    # simply appending prefix is not enough
                    # we fetch actual type name from base poly children
                    stripped_prefix = (
                        current_prefix[:-1]
                        if current_prefix[-1] == "_"
                        else current_prefix
                    )
                    try:
                        typed_child_name = cls.base_poly_children_types[
                            stripped_prefix
                        ][data["type"]]
                        prefixed_key = f"{typed_child_name}_{key}"
                    except KeyError:
                        prefixed_key = f"Cannot find type {data['type']} in {cls.base_poly_children_types[stripped_prefix]}"
                else:
                    prefixed_key = f"{current_prefix}{key}"

                if isinstance(value, dict):
                    # for nested dicts, recursively process with composed prefix
                    flattened = _add_prefix_recursive(cls, value, f"{prefixed_key}_")
                    result.update(flattened)
                else:
                    # leaf value - add it with the full prefix
                    result[prefixed_key] = value

            return result

        return _add_prefix_recursive(cls, d, prefix)

    def _handle_list_config(
        config: Dict[str, Any], prefix: str
    ) -> List[Dict[str, Any]]:
        """Handle special case where config has a '_list' key."""
        list_data = config["_list"]
        if not isinstance(list_data, list):
            return []

        all_exploded = []
        for item in list_data:
            if isinstance(item, dict):
                exploded = _explode_dict_recursive(item)
                all_exploded.extend(exploded)
            else:
                # non-dict items are wrapped
                all_exploded.append({"_value": item})

        return [_add_prefix_to_dict(cls, cfg, prefix) for cfg in all_exploded]

    # handle special case where config has a '_list' key (from load_yaml_config)
    if isinstance(config, dict) and len(config) == 1 and "_list" in config:
        return _handle_list_config(config, prefix)

    # standard case: explode the config and add prefixes
    exploded_configs = _explode_dict_recursive(config)
    return [_add_prefix_to_dict(cls, cfg, prefix) for cfg in exploded_configs]


def topological_sort(dataclass_dependencies: dict) -> list:
    """Topological sort of dataclass dependencies.

    Returns:
        List of dataclass names in topological order.
    """
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
    cli_provided_args: set,
):
    """Overwrite args with values from config.

    Args:
        args: The dictionary of arguments to overwrite. Must be flat.
        config: The dictionary of values to overwrite args with. Can be nested.
        keys_to_file_field_names: A dictionary mapping keys to the file field names that provided them.
        default_values: A dictionary of default values for args.
        cli_provided_args: A set with the argument names that were provided via CLI.
    """

    for key, value in config.items():
        if isinstance(value, dict):
            # a nested config object: we compose the prefix
            overwrite_args_with_config(
                args, value, keys_to_file_field_names, default_values, cli_provided_args
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
    # skip all classes with default None and that have not been provided by the user
    classes_to_skip = set()
    for cls, dependencies in self.dataclass_dependencies.items():
        cls_type_arg = cls + "_type"  # to specify a class, one provides the type
        if (
            cls in self.args_with_default_none
            and cls_type_arg not in self.provided_args
        ):
            classes_to_skip.add(cls)
            for dependency in dependencies:
                classes_to_skip.add(dependency)

    filtered_dependencies = {}
    for cls, dependencies in self.dataclass_dependencies.items():
        if cls not in classes_to_skip:
            filtered_dependencies[cls] = [
                dep for dep in dependencies if dep not in classes_to_skip
            ]

    # list of classes, from the most dependent to the least dependent
    sorted_classes = topological_sort(filtered_dependencies)

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
                if prefixed_field_name in instances:
                    args[original_field_name] = instances[prefixed_field_name]
                else:
                    # if not found in instances, the class has not been provided by the user and is None by default
                    args[original_field_name] = None
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
    Create dataclass instances from CLI arguments and config files.

    Returns:
        List[cls]: A list of instances of the dataclass, one for each combination of configs created.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    all_default_values = {}
    argnames_to_field_names = {}

    # build argument parser from dataclass fields
    for field in fields(cls):
        _add_field_to_parser(
            cls, field, parser, all_default_values, argnames_to_field_names
        )

    args = parser.parse_args()
    cli_provided_args = _get_cli_provided_args(argnames_to_field_names)

    # load and process config files
    loaded_configs = _load_config_files(cls, args)

    # create all combinations of configs
    all_config_combinations, all_keys_to_file_field_names = _create_config_combinations(
        loaded_configs
    )

    # merge cli args with config combinations
    final_args, all_provided_args = _merge_args_with_configs(
        args,
        all_config_combinations,
        all_keys_to_file_field_names,
        all_default_values,
        cli_provided_args,
    )

    return_clss = []
    for i, arg_instance in enumerate(final_args):
        _return_cls = cls(**vars(arg_instance))
        _return_cls.provided_args = all_provided_args[i]
        return_clss.append(_return_cls)

    return return_clss


def _add_field_to_parser(
    cls, field, parser, all_default_values, argnames_to_field_names
):
    """Add a single dataclass field as an argument to the parser."""
    nargs = None
    action = None
    field_type = field.type

    # extract metadata
    help_text = cls.metadata_mapping[field.name].get("help", None)
    argname = cls.metadata_mapping[field.name].get("argname", None)

    # validate argname uniqueness
    if argname in argnames_to_field_names:
        raise ValueError(
            f"Cannot have multiple fields with the same argname: {argname} already exists for field {argnames_to_field_names[argname]}"
        )
    elif argname is not None:
        argnames_to_field_names[argname] = field.name

    # handle optional types
    is_field_optional = is_optional(field.type)
    if is_field_optional:
        field_type = get_inner_type(field.type)

    # configure type-specific parameters
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

    # build argument parameters
    arg_params = {
        "type": field_type,
        "action": action,
        "help": help_text,
    }

    # handle default values
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

    # add argument to parser
    cli_arg_name = field.name.replace("_", "-") if argname is None else argname
    parser.add_argument(f"--{cli_arg_name}", dest=field.name, **arg_params)


def _get_cli_provided_args(argnames_to_field_names) -> Dict[str, Any]:
    """Determine which arguments were explicitly provided via CLI and capture their values.

    Supports the following forms:
        1. --arg=value         -> value is "value"
        2. --arg value         -> value is "value"
        3. --flag              -> value is True (boolean flag)
    """

    cli_provided_args: Dict[str, Any] = {}

    argv = sys.argv
    idx = 1  # skip program name
    while idx < len(argv):
        token = argv[idx]

        # we only care about long-form options that start with "--"
        if not token.startswith("--"):
            idx += 1
            continue

        # strip leading dashes
        option = token[2:]

        # Case 1: --arg=value
        if "=" in option:
            arg_name, arg_value = option.split("=", 1)
        else:
            arg_name = option
            # Case 2 or 3: value may be the next token unless it's another flag
            if idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
                arg_value = argv[idx + 1]
                idx += 1  # skip the value token on next iteration
            else:
                # Case 3: boolean flag with no explicit value
                arg_value = True

        # Map to dataclass field name if argname alias is present
        if arg_name in argnames_to_field_names:
            field_key = argnames_to_field_names[arg_name]
        else:
            # standard conversion: kebab-case -> snake_case
            field_key = arg_name.replace("-", "_")

        cli_provided_args[field_key] = arg_value

        idx += 1

    return cli_provided_args


def _load_config_files(cls, args):
    """Load and process all config files specified in arguments."""
    loaded_configs: Dict[str, List[Dict[str, Any]]] = {}

    logger.info("--------------------------------")
    logger.info("BEGIN LOADING ARGS FROM FILES")
    logger.info("--------------------------------")

    for file_field_name in cls.dataclass_file_fields.values():
        file_path = getattr(args, file_field_name, None)
        if not file_path:
            continue

        file_config = load_yaml_config(file_path)

        # determine prefix for this config file
        # cli args are provided without the root class prefix except for the root _from_file arg
        name_of_class_for_file = file_field_name.replace("_from_file", "").replace(
            "-", ""
        )
        if name_of_class_for_file == cls.root_dataclass_name:
            prefix = ""
        else:
            prefix = f"{name_of_class_for_file}_"

        loaded_configs[file_field_name] = explode_dict(cls, file_config, prefix)

    # log config loading summary
    total_configs = 0
    for file_field_name, configs in loaded_configs.items():
        n_configs = len(configs)
        logger.info(
            f"File field name: {file_field_name}. Expanded to {n_configs} configs."
        )
        total_configs += n_configs

    return loaded_configs


def _create_config_combinations(loaded_configs):
    """Create cartesian product of all loaded configs."""
    all_config_combinations = []
    all_keys_to_file_field_names: List[Dict[str, str]] = []

    if not loaded_configs:
        return all_config_combinations, all_keys_to_file_field_names

    # get config lists for cartesian product
    config_lists = list(loaded_configs.values())
    file_field_names = list(loaded_configs.keys())

    # generate all combinations
    # loaded config dicts are already flattened, so we just need to combine them
    # i.e. {a: [1, 2], b: [3, 4]} -> [{1, 3}, {1, 4}, {2, 3}, {2, 4}] (numbers represent flat dicts)
    for combination in product(*config_lists):
        combined_config = {}
        params_to_files = {}

        for config, current_file_field_name in zip(combination, file_field_names):
            # check for conflicts between configs
            for key, value in config.items():
                if key in combined_config:
                    raise ValueError(
                        f"Arg {key} provided by {current_file_field_name} is also set by {params_to_files[key]}."
                    )
                combined_config[key] = value
                params_to_files[key] = current_file_field_name

        all_config_combinations.append(combined_config)
        all_keys_to_file_field_names.append(params_to_files)

    logger.info(f"Created {len(all_config_combinations)} total config combinations")
    logger.info("--------------------------------")
    logger.info("END LOADING ARGS FROM FILES")
    logger.info("--------------------------------")

    return all_config_combinations, all_keys_to_file_field_names


def _merge_args_with_configs(
    args,
    all_config_combinations,
    all_keys_to_file_field_names,
    all_default_values,
    cli_provided_args,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Merge cli arguments with all config combinations.

    Returns:
        list of all flatclass args and list of user-provided args for each config combination

    Args:
        args: The dictionary of arguments to overwrite. Must be flat.
        all_config_combinations: A list of all combinations of configs.
        all_keys_to_file_field_names: A list of dictionaries mapping keys to the file field names that provided them.
    """
    all_provided_args: List[Dict[str, Any]] = []

    if not all_config_combinations:
        return [args], all_provided_args

    final_args = []
    for config, keys_to_file_field_names in zip(
        all_config_combinations, all_keys_to_file_field_names
    ):
        _provided_args_in_config = {
            **config,
            **cli_provided_args,
        }  # collision prevention is done prior to this
        all_provided_args.append(_provided_args_in_config)

        args_copy = copy.deepcopy(args)
        overwrite_args_with_config(
            args_copy,
            config,
            keys_to_file_field_names,
            all_default_values,
            cli_provided_args,
        )
        final_args.append(args_copy)

    return final_args, all_provided_args


def get_config_class_by_type_name(config_class: Any, type_name: str) -> Any:
    for subclass in get_all_subclasses(config_class):
        if subclass.get_type().name.upper() == type_name.upper():
            return subclass

    raise ValueError(f"Config class with name {type_name} not found.")


def _initialize_dataclass_state():
    """Initialize collections for tracking dataclass metadata during flattening."""
    return {
        "fields_with_defaults": [],
        "fields_without_defaults": [],
        "dataclass_args": defaultdict(list),
        "dataclass_dependencies": defaultdict(
            list
        ),  # maps unique nested dataclass names to their uniquely named dependencies
        "metadata_mapping": {},
        "file_fields": {},  # maps dataclass to its file field name
        "names_to_classes": {},  # maps unique nested dataclass names to their corresponding dataclass class
        "base_poly_children": {},  # maps unique (by name) base poly configs to {children names: children classes} (Dict[str, Dict[str, Any]])
        "base_poly_children_types": {},  # maps unique (by name) base poly configs to {children types: children names} (Dict[str, Dict[str, str]])
        "args_with_default_none": set(),  # the set of args that have a default value of None
    }


def _add_file_argument(state, target_cls, file_field_name: str):
    """Add a '_from_file' cli argument for the given dataclass."""
    state["fields_with_defaults"].append((file_field_name, Optional[str], None))
    state["metadata_mapping"][file_field_name] = {
        "help": f"Path to YAML/JSON configuration file for {target_cls.__name__}."
    }
    state["file_fields"][target_cls] = file_field_name


def _get_field_type_info(field):
    """Extract type information from a dataclass field."""
    if is_optional(field.type):
        return get_inner_type(field.type), True
    return field.type, False


def _get_default_value_for_poly_field(field, field_type):
    """Get the default value for a polymorphic config field."""
    if field.default_factory is not MISSING:
        return str(field.default_factory().get_type())
    elif field.default is not MISSING:
        if field.default is None:
            return "None"
        return str(field.default.get_type())
    else:
        raise ValueError(
            f"Field {field.name} of type {field_type} must have a default or default_factory"
        )


def _handle_polymorphic_config_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass, prefix
):
    """Process a field that is a BasePolyConfig subclass."""
    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    state["base_poly_children"][prefixed_name] = {}
    state["base_poly_children_types"][prefixed_name] = {}

    type_field_name = f"{prefixed_name}_type"
    default_value = _get_default_value_for_poly_field(field, field_type)

    state["fields_with_defaults"].append(
        (type_field_name, type(default_value), default_value)
    )
    state["metadata_mapping"][type_field_name] = field.metadata

    # add _from_file to base poly config with explicit @allow_from_file
    if has_allow_from_file_decorator(field_type):
        file_field_name = f"{prefixed_name}_from_file"
        _add_file_argument(state, field_type, file_field_name)

    # process all subclasses of the polymorphic config
    assert hasattr(field_type, "__dataclass_fields__")
    for subclass in get_all_subclasses(field_type):
        child_name = prefix + to_snake_case(subclass.__name__)
        state["base_poly_children"][prefixed_name][child_name] = subclass
        state["base_poly_children_types"][prefixed_name][
            subclass.get_type().name.lower()
        ] = child_name
        state["dataclass_dependencies"][prefixed_input_dataclass].append(child_name)

        _process_single_dataclass(
            state, subclass, f"{to_snake_case(prefix[:-1] + subclass.__name__)}_"
        )


def _handle_nested_dataclass_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass, prefix
):
    """Process a field that is a nested dataclass."""
    dependency_name = prefix + to_snake_case(field_type.__name__)
    state["dataclass_dependencies"][prefixed_input_dataclass].append(dependency_name)
    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    _process_single_dataclass(
        state, field_type, f"{prefix + to_snake_case(field_type.__name__)}_"
    )


def _handle_primitive_field(
    state, field, field_type, prefixed_name, prefixed_input_dataclass
):
    """Process a field that is a primitive type."""
    field_default = field.default if field.default is not MISSING else MISSING
    field_default_factory = (
        field.default_factory if field.default_factory is not MISSING else MISSING
    )

    if field_default is not MISSING:
        state["fields_with_defaults"].append((prefixed_name, field_type, field_default))
    elif field_default_factory is not MISSING:
        state["fields_with_defaults"].append(
            (prefixed_name, field_type, field_default_factory)
        )
    else:
        state["fields_without_defaults"].append((prefixed_name, field_type))

    state["dataclass_args"][prefixed_input_dataclass].append(
        (prefixed_name, field.name, field_type)
    )
    state["metadata_mapping"][prefixed_name] = field.metadata


def _process_single_dataclass(state, input_dataclass, prefix=""):
    """Process a single dataclass, flattening its fields and handling special cases."""
    prefixed_class_name = (
        f"{prefix[:-1]}" if prefix else f"{to_snake_case(input_dataclass.__name__)}"
    )

    # initialize dependency tracking for this dataclass
    _ = state["dataclass_dependencies"][prefixed_class_name]
    state["names_to_classes"][prefixed_class_name] = input_dataclass

    # add _from_file argument if decorator is present
    if has_allow_from_file_decorator(input_dataclass):
        file_field_name = (
            f"{prefix}from_file"
            if prefix
            else f"{to_snake_case(input_dataclass.__name__)}_from_file"
        )
        _add_file_argument(state, input_dataclass, file_field_name)

    # process each field in the dataclass
    for field in fields(input_dataclass):
        prefixed_name = f"{prefix}{field.name}"
        field_type, _ = _get_field_type_info(field)

        if field.default is None:
            state["args_with_default_none"].add(prefixed_name)

        if is_subclass(field_type, BasePolyConfig):
            _handle_polymorphic_config_field(
                state, field, field_type, prefixed_name, prefixed_class_name, prefix
            )
        elif hasattr(field_type, "__dataclass_fields__"):
            _handle_nested_dataclass_field(
                state, field, field_type, prefixed_name, prefixed_class_name, prefix
            )
        else:
            _handle_primitive_field(
                state, field, field_type, prefixed_name, prefixed_class_name
            )


def _create_flat_class_type(state):
    """Create the final flattened dataclass type with all metadata attached."""
    all_fields = state["fields_without_defaults"] + state["fields_with_defaults"]
    flat_class = make_dataclass("FlatClass", all_fields)

    # attach metadata to the class
    flat_class.dataclass_args = state["dataclass_args"]
    flat_class.dataclass_dependencies = state["dataclass_dependencies"]
    flat_class.dataclass_names_to_classes = state["names_to_classes"]
    flat_class.metadata_mapping = state["metadata_mapping"]
    flat_class.dataclass_file_fields = state["file_fields"]
    flat_class.base_poly_children = state["base_poly_children"]
    flat_class.base_poly_children_types = state["base_poly_children_types"]
    flat_class.args_with_default_none = state["args_with_default_none"]
    return flat_class


def create_flat_dataclass(input_dataclass: Any) -> Any:
    """
    Creates a new FlatClass type by recursively flattening the input dataclass.
    This allows for easy parsing of command line arguments along with storing/loading the configuration to/from a file.
    """
    state = _initialize_dataclass_state()
    _process_single_dataclass(state, input_dataclass)

    flat_class = _create_flat_class_type(state)
    flat_class.root_dataclass_name = to_snake_case(input_dataclass.__name__)

    # attach helper methods
    flat_class.reconstruct_original_dataclass = reconstruct_original_dataclass
    flat_class.create_from_cli_args = create_from_cli_args

    return flat_class
