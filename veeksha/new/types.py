from veeksha.types.base_int_enum import BaseIntEnum


# ----- Traffic -----
class TrafficType(BaseIntEnum):
    RATE = 1
    CONCURRENT = 2


# ----- Interval / length generators -----
class IntervalGeneratorType(BaseIntEnum):
    POISSON = 1
    GAMMA = 2
    FIXED = 3


class LengthGeneratorType(BaseIntEnum):
    ZIPF = 1
    UNIFORM = 2
    FIXED = 3


# ----- Content -----
class SessionGeneratorType(BaseIntEnum):
    SYNTHETIC = 1
    LMEVAL = 2
    TRACE = 3


class TraceFlavorType(BaseIntEnum):
    CLAUDE_CODE = 1
    MOONCAKE_CONV = 2
    RAG = 3


class ChannelModality(BaseIntEnum):
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class SessionGraphType(BaseIntEnum):
    LINEAR = 1


# ----- Evaluation -----
class EvaluationType(BaseIntEnum):
    PERFORMANCE = 1
    ACCURACY = 2


# ----- Client -----
class ClientType(BaseIntEnum):
    OPENAI_CHAT_COMPLETIONS = 1
    OPENAI_COMPLETIONS = 2
    OPENAI_ROUTER = 3


# ----- Server -----
class ServerType(BaseIntEnum):
    VLLM = 1
    VAJRA = 2
    SGLANG = 3


# ----- SLO -----
class SloType(BaseIntEnum):
    CONSTANT = 1
