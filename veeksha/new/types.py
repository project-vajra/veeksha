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
    OPENAI_CHAT = 1
