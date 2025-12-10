from veeksha.types.base_int_enum import BaseIntEnum


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
