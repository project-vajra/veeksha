"""Provider-agnostic client interfaces for the Veeksha framework."""

from veeksha.client.base import BaseLLMClient
from veeksha.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.client.openai_router import OpenAIRouterClient
from veeksha.client.registry import ClientRegistry
from veeksha.client.streaming_tts import StreamingTTSClient
from veeksha.client.stt import STTClient
from veeksha.client.tts import TTSClient

__all__ = [
    "BaseLLMClient",
    "ClientRegistry",
    "OpenAIChatCompletionsClient",
    "OpenAIRouterClient",
    "StreamingTTSClient",
    "STTClient",
    "TTSClient",
]
