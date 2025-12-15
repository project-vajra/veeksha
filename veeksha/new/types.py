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
    TRACE = 2
    LMEVAL = 3


class ChannelModality(BaseIntEnum):
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class SessionGraphType(BaseIntEnum):
    LINEAR = 1
