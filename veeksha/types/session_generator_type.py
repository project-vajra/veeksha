from veeksha.types.base_int_enum import BaseIntEnum


class SessionGeneratorType(BaseIntEnum):
    TRACE_SYNTHETIC = 1 # generate sessions based on trace file
    TRACE = 2 # use sessions from trace file as is
