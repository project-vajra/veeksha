import json
import yaml
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from typing import Any, Optional, get_args, Dict, List

from veeksha.logger import init_logger
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
    to_snake_case,
    load_yaml_config,
)

logger = init_logger(__name__)


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


def overwrite_args_with_file_config(args: dict, file_config: dict, file_field_name: str, default_values: dict, prefix: str = ""):
    """ Overwrite args with values from file_config in a DFS manner """
        
    for key, value in file_config.items():
        if "type" in file_config and key != "type":
            # i.e. request_generator_config -> trace_request_generator_config
            key = f"{file_config['type']}_{prefix}{key}"
        else:
            key = f"{prefix}{key}"

        if isinstance(value, dict):
            # a nested config object
            # Pass the already-composed key as the new prefix to avoid duplicating the old prefix.
            overwrite_args_with_file_config(args, value, file_field_name, default_values, f"{key}_")
            continue
        
        # Ignore keys that are not recognised by the FlatClass.
        if not hasattr(args, key):
            logger.warning(f"Arg {key} provided by {file_field_name} not found in supported args.")
            continue
        
        if getattr(args, key) == default_values.get(key):
            print(f"Overwriting {key} with {value}")
            setattr(args, key, value)
        else:
            logger.warning(f"Arg {key} provided by {file_field_name} set via CLI. Skipped overwrite.")


def reconstruct_original_dataclass(self) -> Any:
    """
    This function is dynamically mapped to FlatClass as an instance method.
    Reconstructs the original dataclass from the flattened representation.
    """
    # list of classes, from the most dependent to the least dependent
    sorted_classes = topological_sort(self.dataclass_dependencies)
    
    print("--------------------------------")
    print("Dataclass dependencies")
    print("--------------------------------")
    for cls, dependencies in self.dataclass_dependencies.items():
        print(cls, [dep for dep in dependencies])
    print("END DATACLASS DEPENDENCIES--------------------------------")
    
    # print(sorted_classes)
    # print("--------------------------------")
    # print("Sorted classes")
    # print("--------------------------------")
    # for cls in sorted_classes:
    #     print(cls)
    # print("END SORTED CLASSES--------------------------------")

    instances = {}

    # iter over classes from least dependent to most
    for _cls in reversed(sorted_classes):
        args = {}
        print(f"Instantiating {_cls}")

        # instantiate current class fields
        for prefixed_field_name, original_field_name, field_type in self.dataclass_args[
            _cls
        ]:
            print("prefixed_field_name", prefixed_field_name)
            print("original_field_name", original_field_name)
            print("field_type", field_type)
            print("--------------------------------")
            if is_subclass(field_type, BasePolyConfig):
                config_type = getattr(self, f"{prefixed_field_name}_type")
                # find all subclasses of field_type and check which one matches the config_type
                config_type_matched = False
                # base poly children cointains all subclasses of the base poly config
                for child_name, child_cls in self.base_poly_children[prefixed_field_name].items():
                    print(f"Checking {child_name} with type {child_cls.get_type()}")
                    if str(child_cls.get_type()) == config_type:
                        config_type_matched = True
                        args[original_field_name] = instances[child_name]
                        print(f"Assigned {original_field_name} to {instances[child_name]}")
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

    print("--------------------------------")
    print(f"Root: {instances[sorted_classes[0]]}")
    return instances[sorted_classes[0]]


@classmethod
def create_from_cli_args(cls) -> Any:
    """
    This function is dynamically mapped to FlatClass as a class method.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    all_default_values = {}

    for field in fields(cls):
        nargs = None
        action = None
        field_type = field.type
        help_text = cls.metadata_mapping[field.name].get("help", None)
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
            all_default_values[field.name] = object() # sentinel value
            if is_field_optional:
                arg_params["default"] = None
            else:
                arg_params["required"] = True

        if nargs:
            arg_params["nargs"] = nargs
        cli_arg_name = field.name.replace("_", "-")
        parser.add_argument(f"--{cli_arg_name}", dest=field.name, **arg_params)

    args = parser.parse_args()

    print(f"default_values: {all_default_values}")

    # load config files and overwrite fields not set via CLI
    for file_field_name in cls.dataclass_file_fields.values():
        file_path = getattr(args, file_field_name, None)
        if not file_path:
            continue
        file_config = load_yaml_config(file_path)

        print(f"file_config for {file_field_name}: {file_config}")

        # todo handle collisions for:
        # - multiple configs are provided
        # - CLI args and file args are provided for the same field
        overwrite_args_with_file_config(args, file_config, file_field_name, all_default_values)

    # # inspect args
    # print("--------------------------------")
    # print("CLI arguments (after merging *_from_file)")
    # print("--------------------------------")
    # for arg_name, arg_value in vars(args).items():
    #     print(arg_name, arg_value)

    instance = cls(**vars(args))

    # # inspect instance 2
    # print("--------------------------------")
    # print("FlatClass 2")
    # print("--------------------------------")
    # for field in fields(instance):
    #     print(field.name, getattr(instance, field.name))

    return instance


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
    dataclass_dependencies = defaultdict(list)  # maps unique nested dataclass names to their uniquely named dependencies
    metadata_mapping = {}
    dataclass_file_fields = {}  # maps dataclass to its file field name
    dataclass_names_to_classes = {}  # maps unique nested dataclass names to their corresponding dataclass class
    base_poly_children: Dict[str, Dict[str, Any]] = {}  # maps unique (by name) base poly configs to their children

    def add_file_arg(target_cls, file_field_name: str):
        # adding the implicit "<prefix>_from_file" CLI flag.
        meta_fields_with_defaults.append((file_field_name, Optional[str], None))
        metadata_mapping[file_field_name] = {
            "help": f"Path to YAML/JSON configuration file for {target_cls.__name__}."
        }
        dataclass_file_fields[target_cls] = file_field_name

    def process_dataclass(_input_dataclass, prefix=""):
        """ Creates a flattened representation of the input dataclass, adding _from_file and _type fields in some cases and populating metadata fields """
        prefixed_input_dataclass = f"{prefix[:-1]}" if prefix else f"{to_snake_case(input_dataclass.__name__)}"
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
                    base_poly_children[prefixed_name][prefix + to_snake_case(subclass.__name__)] = subclass
                    dataclass_dependencies[prefixed_input_dataclass].append(prefix + to_snake_case(subclass.__name__))
                    process_dataclass(subclass, f"{to_snake_case(prefix[:-1] + subclass.__name__)}_")
                continue
            # if field is a dataclass, recursively process it
            if hasattr(field_type, "__dataclass_fields__"):
                dataclass_dependencies[prefixed_input_dataclass].append(prefix + to_snake_case(field_type.__name__))
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
    
    print("--------------------------------")
    print("Dataclass args")
    print("--------------------------------")
    for cls, args in dataclass_args.items():
        print(cls, [arg for arg in args])
    print("END DATACLASS ARGS--------------------------------")
    
    print("--------------------------------")
    print("Dataclass names to classes")
    print("--------------------------------")
    for cls_name, cls in dataclass_names_to_classes.items():
        print(cls_name, cls)
    print("END DATACLASS NAMES TO CLASSES--------------------------------")
    
    # print("--------------------------------")
    # print("Dataclass names to classes")
    # print("--------------------------------")
    # for cls_name, cls in dataclass_names_to_classes.items():
    #     print(cls_name, cls)
    # print("END DATACLASS NAMES TO CLASSES--------------------------------")
    
    # # inspect dataclass dependencies
    # print("--------------------------------")
    # print("Dataclass dependencies")
    # print("--------------------------------")
    # for cls, dependencies in dataclass_dependencies.items():
    #     print(cls, [dep for dep in dependencies])
    # print("--------------------------------")
    # print("Dataclass args")
    # print("--------------------------------")
    # for cls, args in dataclass_args.items():
    #     print(cls, [arg for arg in args])
        
    # print("END DATACLASS ARGS--------------------------------")

    meta_fields = meta_fields_without_defaults + meta_fields_with_defaults
    
    print("--------------------------------")
    print("Base poly children")
    print("--------------------------------")
    for cls, children in base_poly_children.items():
        print(cls, [f"{child_name}: {child_class}" for child_name, child_class in children.items()])
    print("END BASE POLY CHILDREN--------------------------------")

    # # inspect meta fields
    # print("--------------------------------")
    # print("Meta fields")
    # print("--------------------------------")
    # for field in meta_fields:
    #     print(field)
    # print("END META FIELDS--------------------------------")

    # Flat dataclass with all default values (unitialized)
    FlatClass = make_dataclass("FlatClass", meta_fields)

    # Metadata fields
    FlatClass.dataclass_args = dataclass_args
    FlatClass.dataclass_dependencies = dataclass_dependencies
    FlatClass.dataclass_names_to_classes = dataclass_names_to_classes
    FlatClass.metadata_mapping = metadata_mapping
    FlatClass.dataclass_file_fields = dataclass_file_fields
    FlatClass.base_poly_children = base_poly_children
    # Helper methods
    FlatClass.reconstruct_original_dataclass = reconstruct_original_dataclass
    FlatClass.create_from_cli_args = create_from_cli_args

    return FlatClass
