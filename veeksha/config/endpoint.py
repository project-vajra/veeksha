from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass(allow_from_file=True)
class EndpointConfig:
    """Configuration for a single API endpoint.

    This class encapsulates the URL, authentication key, and optional name
    for an API endpoint to be benchmarked.
    """

    name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional name for the endpoint (useful for identification in logs/results)."
        },
    )
    api_url: str = field(
        default="http://localhost:8000/v1",
        metadata={"help": "The API URL for the endpoint."},
    )
    api_key: str = field(
        default="token-abc123",
        metadata={"help": "The API key for authentication."},
    )
