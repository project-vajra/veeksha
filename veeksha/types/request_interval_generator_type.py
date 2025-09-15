from enum import Enum


class RequestIntervalGeneratorType(Enum):
    POISSON = "poisson"
    GAMMA = "gamma"
    STATIC = "static"
    TRACE = "trace"
