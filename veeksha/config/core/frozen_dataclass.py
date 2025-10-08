import dataclasses
import typing


@typing.dataclass_transform()
def frozen_dataclass(_cls=None, **kwargs):
    """
    A decorator that creates a frozen dataclass, allowing attribute modifications
    only during the __post_init__ method.

    Args:
        _cls: The class to decorate (for decorator syntax handling).
        **kwargs: Additional keyword arguments to pass to dataclasses.dataclass.
        allow_from_file (bool): If True, mark the dataclass as allowing configuration
            to be loaded from a file.
    """

    # Extract custom keyword arguments handled by this decorator
    allow_from_file = kwargs.pop("allow_from_file", False)

    def wrap(cls):
        # Apply dataclass with frozen=True
        datacls = dataclasses.dataclass(cls, frozen=True, **kwargs)  # type: ignore

        # Mark as allowing file-based configuration if requested
        if allow_from_file:
            setattr(datacls, "_allow_from_file", True)

        # Store the original __post_init__ if it exists
        original_post_init = getattr(cls, "__post_init__", None)

        # Define a new __post_init__ method
        def __post_init__(self):
            # Set the flag to allow attribute modifications
            object.__setattr__(self, "_in_post_init", True)
            try:
                if original_post_init:
                    original_post_init(self)
            finally:
                # Reset the flag after __post_init__ completes
                object.__setattr__(self, "_in_post_init", False)

        datacls.__post_init__ = __post_init__

        # Store the original __setattr__ method
        original_setattr = datacls.__setattr__

        # Define a new __setattr__ method
        def __setattr__(self, name, value):
            # Check if we are in __post_init__
            if getattr(self, "_in_post_init", False):
                object.__setattr__(self, name, value)
            else:
                original_setattr(self, name, value)

        datacls.__setattr__ = __setattr__

        # Define pickle support methods
        def __getstate__(self):
            # Return only the dataclass fields for pickling
            return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

        datacls.__getstate__ = __getstate__

        def __setstate__(self, state):
            # Restore from pickled state
            object.__setattr__(self, "_in_post_init", True)
            for key, value in state.items():
                object.__setattr__(self, key, value)
            # Call __post_init__ to initialize derived attributes
            if hasattr(self, '__post_init__'):
                self.__post_init__()
            else:
                # If no __post_init__, just clear the flag
                object.__setattr__(self, "_in_post_init", False)

        datacls.__setstate__ = __setstate__

        return datacls

    # Support both @frozen_dataclass and @frozen_dataclass() syntax
    if _cls is None:
        return wrap
    return wrap(_cls)
