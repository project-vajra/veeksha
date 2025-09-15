from veeksha.types.base_int_enum import BaseIntEnum


class LMEvalOutputType(BaseIntEnum):
    LOGLIKELIHOOD = 1
    LOGLIKELIHOOD_ROLLING = 2
    GENERATE_UNTIL = 3
    MULTIPLE_CHOICE = 4
