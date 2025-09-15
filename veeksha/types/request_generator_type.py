from enum import Enum


class RequestGeneratorType(Enum):
    SYNTHETIC = "synthetic"
    TRACE = "trace"
    LMEVAL = "lmeval"
