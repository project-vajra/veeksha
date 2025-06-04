import hashlib
from dataclasses import fields, is_dataclass
from typing import Union, get_args, get_origin

primitive_types = {int, str, float, bool, type(None)}


def _get_hash(key):
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def get_all_subclasses(cls):
    subclasses = cls.__subclasses__()
    return subclasses + [g for s in subclasses for g in get_all_subclasses(s)]


def is_primitive_type(field_type: type) -> bool:
    # Check if the type is a primitive type
    return field_type in primitive_types


def is_generic_composed_of_primitives(field_type: type) -> bool:
    origin = get_origin(field_type)
    if origin in {list, dict, tuple, Union}:
        # Check all arguments of the generic type
        args = get_args(field_type)
        return all(is_composed_of_primitives(arg) for arg in args)
    return False


def is_composed_of_primitives(field_type: type) -> bool:
    # Check if the type is a primitive type
    if is_primitive_type(field_type):
        return True

    # Check if the type is a generic type composed of primitives
    if is_generic_composed_of_primitives(field_type):
        return True

    return False


def to_snake_case(name: str) -> str:
    return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip("_")


def is_optional(field_type: type) -> bool:
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def is_list(field_type: type) -> bool:
    # Check if the field type is a List
    return get_origin(field_type) is list


def is_dict(field_type: type) -> bool:
    # Check if the field type is a Dict
    return get_origin(field_type) is dict


def is_bool(field_type: type) -> bool:
    return field_type is bool


def get_inner_type(field_type: type) -> type:
    return next(t for t in get_args(field_type) if t is not type(None))


def is_subclass(cls, parent: type) -> bool:
    return hasattr(cls, "__bases__") and parent in cls.__bases__


def dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif is_dataclass(obj):
        data = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            data[field.name] = dataclass_to_dict(value)
        # Include members created in __post_init__
        for key, value in obj.__dict__.items():
            if key not in data:
                data[key] = dataclass_to_dict(value)
        # Include the name of the class
        if hasattr(obj, "get_type") and callable(getattr(obj, "get_type")):
            data["name"] = str(obj.get_type())  # type: ignore
        elif hasattr(obj, "get_name") and callable(getattr(obj, "get_name")):
            data["name"] = obj.get_name()  # type: ignore
        return data
    else:
        return obj


def dict_to_args(class_dict):
    args = []
    for key, value in class_dict.items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
                else:
                    args.append(f"--no-{key}")
            else:
                args.append(f"--{key} {value}")
    return " ".join(args)