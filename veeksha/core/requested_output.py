"""Requested output specifications for benchmarks.

This module defines data structures for specifying what output is expected
from the model for each request. This is separate from input content generation
to support cross-modal benchmarks (e.g., text-to-image, image-to-text).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TextOutputSpec:
    """Specification for requested text output.

    Attributes:
        target_tokens: Target number of tokens for the model to generate.
    """

    target_tokens: int


@dataclass
class ImageOutputSpec:
    """Specification for requested image output.

    Attributes:
        num_images: Number of images to generate per request.
        size: Image dimensions (e.g., "1024x1024").
        quality: Image quality setting (e.g., "hd", "standard").
    """

    num_images: int = 1
    size: Optional[str] = None
    quality: Optional[str] = None


@dataclass
class AudioOutputSpec:
    """Specification for requested audio output."""


@dataclass
class VideoOutputSpec:
    """Specification for requested video output."""


@dataclass
class RequestedOutputSpec:
    """Container for all requested output specifications.

    This allows specifying expected output for multiple modalities
    simultaneously. Only populate the fields relevant to your benchmark.

    For example:
    - Text-to-text: Only `text` field populated
    - Text-to-image: Both `text` (for input) and `image` (for output)
    - Multi-modal output: Multiple fields populated

    Attributes:
        text: Specification for text output, if expected.
        image: Specification for image output, if expected.
        audio: Specification for audio output, if expected.
        video: Specification for video output, if expected.
    """

    text: Optional[TextOutputSpec] = None
    image: Optional[ImageOutputSpec] = None
    audio: Optional[AudioOutputSpec] = None
    video: Optional[VideoOutputSpec] = None
