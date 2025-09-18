from dataclasses import field
from typing import List

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generators.interval_generator.base_generator import (
    BaseRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.request_generator.base_generator import (
    BaseRequestGeneratorConfig,
)
from veeksha.types.request_generator_type import RequestGeneratorType


@frozen_dataclass
class LmevalRequestGeneratorConfig(BaseRequestGeneratorConfig):
    tasks: List[str] = field(
        default_factory=lambda: ["hellaswag"],
        metadata={"help": "The tasks to evaluate the language model on."},
    )
    num_fewshot: int = field(
        default=1,
        metadata={"help": "The number of fewshot examples to use for the tasks."},
    )
    limit: int = field(
        default=10,
        metadata={"help": "The number of examples to evaluate on."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig
    )

    def __post_init__(self):
        if not self.tasks:
            raise ValueError("LMEvalRequestGenerator requires at least one task.")

    @classmethod
    def get_type(cls):
        return RequestGeneratorType.LMEVAL
