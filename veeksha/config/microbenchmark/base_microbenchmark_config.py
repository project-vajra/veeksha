from dataclasses import field
from typing import Optional

from veeksha.constants.configuration_constants import DEFAULT_SEED
from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class BaseMicrobenchmarkConfig(BasePolyConfig):
    """
    Base class for microbenchmarks (prefill, decode, mixed batching, etc.)
    """

    model: str = field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        metadata={"help": "The model to use for this microbenchmark."},
    )
    api_url: str = field(
        default="http://localhost:30000/v1",
        metadata={"help": "The API URL for the benchmark."},
    )
    api_key: str = field(
        default="token-abc123",
        metadata={"help": "The API key for the benchmark."},
    )
    tokenizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer to use for this microbenchmark. By default, the tokenizer is inferred from the model."
        },
    )
    additional_sampling_params: str = field(
        default="{}",
        metadata={"help": "Additional sampling params."},
    )
    timeout: int = field(
        default=1200,
        metadata={"help": "The amount of time to run each profiling run for."},
    )
    seed: int = field(
        default=DEFAULT_SEED,
        metadata={"help": "Seed for the random number generator."},
    )
    output_dir: str = field(
        default="microbenchmark_experiments",
        metadata={"help": "Output directory for microbenchmark results."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Wandb project."},
    )
