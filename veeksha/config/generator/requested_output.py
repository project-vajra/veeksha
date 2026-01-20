"""Configuration classes for output specifications.

This module defines config classes for specifying expected output from the model.
Output specifications are separate from input channel generators to support
cross-modal benchmarks (e.g., text-to-image, image-to-text).
"""

from dataclasses import field
from typing import Optional

from veeksha.config.core.frozen_dataclass import frozen_dataclass
from veeksha.config.generator.length import (
    BaseLengthGeneratorConfig,
    UniformLengthGeneratorConfig,
)
from veeksha.types import LengthGeneratorType


@frozen_dataclass
class TextOutputSpecConfig:
    """Configuration for expected text output.

    Attributes:
        output_length_generator: Generator for the target number of output tokens.
    """

    output_length_generator: BaseLengthGeneratorConfig = field(
        default_factory=UniformLengthGeneratorConfig,
        metadata={
            "help": f"The generator for the output length. {LengthGeneratorType.help_str()}"
        },
    )


@frozen_dataclass
class ImageOutputSpecConfig:
    """Configuration for expected image output.

    Attributes:
        num_images: Number of images to generate per request.
        size: Image dimensions (e.g., "1024x1024").
        quality: Image quality setting (e.g., "hd", "standard").
    """

    num_images: int = field(
        default=1,
        metadata={"help": "Number of images to generate per request."},
    )
    size: Optional[str] = field(
        default=None,
        metadata={"help": "Image dimensions (e.g., '1024x1024')."},
    )
    quality: Optional[str] = field(
        default=None,
        metadata={"help": "Image quality setting (e.g., 'hd', 'standard')."},
    )


@frozen_dataclass
class AudioOutputSpecConfig:
    """Configuration for expected audio output."""


@frozen_dataclass
class VideoOutputSpecConfig:
    """Configuration for expected video output."""


@frozen_dataclass
class OutputSpecConfig:
    """Container for all output specification configurations.

    This allows specifying expected output for multiple modalities
    simultaneously. Only populate the fields relevant to your benchmark.

    Example YAML:
        output_spec:
          text:
            output_length_generator:
              type: uniform
              min: 100
              max: 500
          # image:
          #   num_images: 3
          #   size: "1024x1024"

    Attributes:
        text: Configuration for text output, if expected.
        image: Configuration for image output, if expected.
        audio: Configuration for audio output, if expected.
        video: Configuration for video output, if expected.
    """

    text: Optional[TextOutputSpecConfig] = field(
        default_factory=TextOutputSpecConfig,
        metadata={"help": "Configuration for expected text output."},
    )
    image: Optional[ImageOutputSpecConfig] = field(
        default=None,
        metadata={"help": "Configuration for expected image output."},
    )
    audio: Optional[AudioOutputSpecConfig] = field(
        default=None,
        metadata={"help": "Configuration for expected audio output."},
    )
    video: Optional[VideoOutputSpecConfig] = field(
        default=None,
        metadata={"help": "Configuration for expected video output."},
    )
