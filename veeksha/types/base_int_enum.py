from enum import IntEnum


class BaseIntEnum(IntEnum):
    def __str__(self):
        return self.name.lower()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]

    @classmethod
    def help_str(cls):
        return f"Available: {', '.join([m.name.lower() for m in cls])}."
