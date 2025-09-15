from enum import Enum


class RequestLengthGeneratorType(Enum):
    UNIFORM = "uniform"
    ZIPF = "zipf"
    TRACE = "trace"
    FIXED = "fixed"
