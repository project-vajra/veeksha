from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass


@frozen_dataclass
class BaseMicrobenchmarkProbeConfig(BasePolyConfig):
    """Base class for microbenchmark probes (prefill, decode, ...)."""
