from veeksha.core.lazy_loader import _LazyLoader
from veeksha.new.types import ClientType
from veeksha.types.base_registry import BaseRegistry


class ClientRegistry(BaseRegistry):
    @classmethod
    def get_key_from_str(cls, key_str: str) -> ClientType:
        return ClientType.from_str(key_str)  # type: ignore


ClientRegistry.register(
    ClientType.OPENAI_CHAT,
    _LazyLoader(
        "veeksha.new.client.openai_chat",
        "OpenAIChatClient",
    ),
)
