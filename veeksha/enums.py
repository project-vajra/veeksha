from enum import Enum



class APIType(Enum):
    COMPLETION = "COMPLETION"
    CHAT_COMPLETION = "CHAT_COMPLETION"


class RequestGeneratorType(Enum):
    SYNTHETIC = "SYNTHETIC"
    TRACE = "TRACE"
    PREFIX = "PREFIX"
    LMEVAL = "LMEVAL"


class LMEvalOutputType(Enum):
    LOGLIKELIHOOD = "LOGLIKELIHOOD"
    LOGLIKELIHOOD_ROLLING = "LOGLIKELIHOOD_ROLLING"
    GENERATE_UNTIL = "GENERATE_UNTIL"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"


class RequestIntervalGeneratorType(Enum):
    POISSON = "POISSON"
    GAMMA = "GAMMA"
    STATIC = "STATIC"
    TRACE = "TRACE"


class RequestLengthGeneratorType(Enum):
    UNIFORM = "UNIFORM"
    ZIPF = "ZIPF"
    TRACE = "TRACE"
    FIXED = "FIXED"
