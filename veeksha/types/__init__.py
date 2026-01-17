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
    FIXED_STAIR = 4


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
    SINGLE_REQUEST = 2
    BRANCHING = 3


# ----- Evaluation -----
class EvaluationType(BaseIntEnum):
    PERFORMANCE = 1
    ACCURACY_LMEVAL = 2


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


# ----- SLOs -----
class SloType(BaseIntEnum):
    CONSTANT = 1


# ----- LMEval -----
class LMEvalOutputType(BaseIntEnum):
    LOGLIKELIHOOD = 1
    LOGLIKELIHOOD_ROLLING = 2
    GENERATE_UNTIL = 3
    MULTIPLE_CHOICE = 4
