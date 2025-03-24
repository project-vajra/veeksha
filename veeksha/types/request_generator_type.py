from veeksha.types.base_int_enum import BaseIntEnum


class RequestGeneratorType(BaseIntEnum):
    SYNTHETIC = 1
    TRACE = 2
    PREFIX = 3
    LMEVAL = 4
