from dataclasses import field

from veeksha.config.core.base_poly_config import BasePolyConfig
from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generator.interval import (
    BaseIntervalGeneratorConfig,
    PoissonIntervalGeneratorConfig,
)
from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
    FixedLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.types import (
    IntervalGeneratorType,
    LengthGeneratorType,
    SessionGraphType,
)


@frozen_dataclass
class BaseSessionGraphGeneratorConfig(BasePolyConfig):
    pass


@frozen_dataclass
class LinearSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of linear request graphs (a sequence of requests).
    """

    inherit_history: bool = field(
        default=True,
        metadata={
            "help": "Whether subsequent requests can inherit history from previous ones."
        },
    )
    num_request_generator: BaseLengthGeneratorConfig = field(
        default_factory=UniformLengthGeneratorConfig,
        metadata={
            "help": f"The generator for the number of requests. {LengthGeneratorType.help_str()}"
        },
    )
    request_wait_generator: BaseIntervalGeneratorConfig = field(
        default_factory=PoissonIntervalGeneratorConfig,
        metadata={
            "help": f"The generator for the wait time between requests. {IntervalGeneratorType.help_str()}"
        },
    )

    @classmethod
    def get_type(cls):
        return SessionGraphType.LINEAR


@frozen_dataclass
class SingleRequestSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of single request graphs (a single request).
    """

    @classmethod
    def get_type(cls):
        return SessionGraphType.SINGLE_REQUEST


@frozen_dataclass
class BranchingSessionGraphGeneratorConfig(BaseSessionGraphGeneratorConfig):
    """
    Generator of branching session graphs (DAGs with fan-out and fan-in).

    Creates layered graphs where each layer can have variable width,
    and nodes connect forward with optional skip connections.

    **Graph structure**:
    The generator builds graphs layer by layer:
    - Layer 0: Root layer (1 node if single_root=True, else sampled width)
    - Layers 1 to N-1: Each layer has independently sampled width

    **Forward pass (fan_out_generator)**:
    Each node creates outgoing edges based on sampled fan_out value.
    Edges target nodes in layer (current + connection_dist) or later.
    Skip connections (connection_dist > 1) are sampled and allow edges to skip layers.

    **Backward pass (fan_in_generator)**:
    Ensures each non-root node has at least the sampled fan_in incoming edges.
    If fan_in exceeds available parent nodes, it is capped to max available.

    **History inheritance**:
    When inherit_history=True, exactly one parent per node is randomly
    selected as the "history parent" whose conversation context is inherited.

    **Edge cases**:
    - fan_out > target layer size: Edges are deduplicated (only one edge per src-dst pair)
    - fan_in > previous layer size: Capped to available nodes (no error)
    - connection_dist > remaining layers: Capped to last layer
    - layer_width samples 0: Not possible (minimum enforced by LengthGenerator)
    """

    num_layers_generator: BaseLengthGeneratorConfig = field(
        default_factory=lambda: UniformLengthGeneratorConfig(min=2, max=5),
        metadata={
            "help": f"Generator for the number of layers (depth). {LengthGeneratorType.help_str()}"
        },
    )
    layer_width_generator: BaseLengthGeneratorConfig = field(
        default_factory=lambda: UniformLengthGeneratorConfig(min=1, max=3),
        metadata={
            "help": f"Generator for width per layer (sampled independently for each layer). {LengthGeneratorType.help_str()}"
        },
    )
    fan_out_generator: BaseLengthGeneratorConfig = field(
        default_factory=lambda: UniformLengthGeneratorConfig(min=1, max=2),
        metadata={
            "help": f"Generator for number of outgoing edges per node. {LengthGeneratorType.help_str()}"
        },
    )
    fan_in_generator: BaseLengthGeneratorConfig = field(
        default_factory=lambda: UniformLengthGeneratorConfig(min=1, max=2),
        metadata={
            "help": f"Generator for minimum incoming edges per node. Capped to available parent nodes. {LengthGeneratorType.help_str()}"
        },
    )
    connection_dist_generator: BaseLengthGeneratorConfig = field(
        default_factory=lambda: FixedLengthGeneratorConfig(value=1),
        metadata={
            "help": f"Generator for forward skip distance (1 = next layer, 2 = skip one layer, etc.). Capped to last layer. {LengthGeneratorType.help_str()}"
        },
    )
    request_wait_generator: BaseIntervalGeneratorConfig = field(
        default_factory=PoissonIntervalGeneratorConfig,
        metadata={
            "help": f"Generator for wait time after all parents complete. {IntervalGeneratorType.help_str()}"
        },
    )
    inherit_history: bool = field(
        default=True,
        metadata={
            "help": "When true, one parent per node is randomly selected as history provider."
        },
    )
    single_root: bool = field(
        default=True,
        metadata={
            "help": "Force layer 0 to have exactly 1 node (typical for chat sessions)."
        },
    )

    @classmethod
    def get_type(cls):
        return SessionGraphType.BRANCHING
