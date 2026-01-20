from abc import ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseChannelRequestContent(ABC):
    """Base class for channel request content."""


@dataclass
class TextChannelRequestContent(BaseChannelRequestContent):
    """Content for a text channel request.

    Attributes:
        input_text: The actual text content to send to the model.
        target_prompt_tokens: The target number of prompt tokens (for tracing/metrics).
    """

    input_text: str
    target_prompt_tokens: Optional[int] = None


@dataclass
class ImageChannelRequestContent(BaseChannelRequestContent):
    """Content for an image channel request.

    Attributes:
        input_image: The actual image content to send to the model.
    """

    input_image: str


@dataclass
class AudioChannelRequestContent(BaseChannelRequestContent):
    """Content for an audio channel request.

    Attributes:
        input_audio: The actual audio content to send to the model.
    """

    input_audio: str


@dataclass
class VideoChannelRequestContent(BaseChannelRequestContent):
    """Content for a video channel request.

    Attributes:
        input_video: The actual video content to send to the model.
    """

    input_video: str
