from abc import ABC
from enum import Enum
from typing import Any, ClassVar, Dict


class BaseRegistry(ABC):
    """
    A generic registry base class. Subclasses will have their own registry dictionary.
    Each subclass can store implementations keyed by an Enum.
    """

    # For MyPy to recognize that all subclasses have a _registry class attribute,
    # declare it here as a ClassVar (class-level variable).
    _registry: ClassVar[Dict[Enum, Any]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own empty registry.
        cls._registry = {}

    @classmethod
    def register(cls, key: Enum, implementation_class: Any) -> None:
        if key in cls._registry:
            return

        cls._registry[key] = implementation_class

    @classmethod
    def unregister(cls, key: Enum) -> None:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        del cls._registry[key]

    @classmethod
    def get(cls, key: Enum, *args: Any, **kwargs: Any) -> Any:
        if key not in cls._registry:
            raise ValueError(f"{key} is not registered")

        return cls._registry[key](*args, **kwargs)
