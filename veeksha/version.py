import warnings
from importlib import import_module
from typing import Any

try:
    _version_module: Any = import_module("veeksha._version")
    __version__ = str(_version_module.__version__)
    __version_tuple__ = tuple(_version_module.__version_tuple__)
except Exception as error:
    warnings.warn(
        f"Failed to read commit hash:\n{error}",
        RuntimeWarning,
        stacklevel=2,
    )

    __version__ = "dev"
    __version_tuple__ = (0, 0, __version__)
