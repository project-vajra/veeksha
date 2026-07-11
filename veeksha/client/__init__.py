"""LLM clients for the new Veeksha framework."""

from veeksha.client.base import BaseLLMClient
from veeksha.client.openai_chat import OpenAIChatCompletionsClient
from veeksha.client.openai_router import OpenAIRouterClient
from veeksha.client.realtime_tts import RealtimeTTSClient
from veeksha.client.registry import ClientRegistry
from veeksha.client.tts import TTSClient

__all__ = [
    "BaseLLMClient",
    "OpenAIChatCompletionsClient",
    "OpenAIRouterClient",
    "RealtimeTTSClient",
    "TTSClient",
    "ClientRegistry",
]
