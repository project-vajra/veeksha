from typing import Any


class _LazyLoader:
    """Lazy loader that defers imports until the class is actually needed."""

    def __init__(self, import_path: str, class_name: str):
        # store import path and class name to import when actually used
        self.import_path = import_path
        self.class_name = class_name
        self._cached_class = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Load the class if needed and instantiate it."""
        if self._cached_class is None:
            # Import finally happens when first accessed
            module = __import__(self.import_path, fromlist=[self.class_name])
            self._cached_class = getattr(module, self.class_name)
        return self._cached_class(*args, **kwargs)
