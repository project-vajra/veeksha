def allow_from_file(cls):
    """
    Decorator to mark a dataclass as allowing file-based configuration.

    Classes decorated with @allow_from_file will have an implicit _from_file argument
    added to their flat representation, allowing them to be instantiated via YAML/JSON.

    Args:
        cls: The dataclass to decorate

    Returns:
        The decorated class with _allow_from_file attribute set to True
    """
    cls._allow_from_file = True
    return cls


def has_allow_from_file_decorator(cls) -> bool:
    """
    Check if a class has the @allow_from_file decorator.
    Only return True if the attribute is defined directly on cls (i.e. not inherited)

    Args:
        cls: The class to check

    Returns:
        True if the class has the @allow_from_file decorator, False otherwise
    """
    return vars(cls).get("_allow_from_file", False)
