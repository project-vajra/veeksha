from enum import Enum


class LMEvalOutputType(Enum):
    LOGLIKELIHOOD = "loglikelihood"
    LOGLIKELIHOOD_ROLLING = "loglikelihood_rolling"
    GENERATE_UNTIL = "generate_until"
    MULTIPLE_CHOICE = "multiple_choice"
