from typing import List

from veeksha.config.generator.session import TraceSessionGeneratorConfig
from veeksha.core.seeding import SeedManager
from veeksha.core.session import Session
from veeksha.core.tokenizer import TokenizerProvider
from veeksha.generator.session.base import BaseSessionGenerator
from veeksha.generator.session.trace.registry import TraceFlavorGeneratorRegistry


class TraceSessionGenerator(BaseSessionGenerator):
    """Trace session generator that delegates to flavor-specific implementations.

    This is the unified entry point for all trace-driven session generation.
    It inspects the flavor config and delegates to the appropriate implementation.
    """

    def __init__(
        self,
        config: TraceSessionGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_provider: TokenizerProvider,
    ):
        super().__init__(config, seed_manager)
        self.config = config

        self._impl = TraceFlavorGeneratorRegistry.get(
            config.flavor.get_type(),
            config,
            config.flavor,
            seed_manager,
            tokenizer_provider,
        )

    def generate_session(self) -> Session:
        """Delegate to flavor implementation."""
        return self._impl.generate_session()

    def capacity(self) -> int:
        """Delegate to flavor implementation."""
        return self._impl.capacity()

    def get_warmup_sessions(self) -> List[Session]:
        """Delegate to flavor implementation."""
        return self._impl.get_warmup_sessions()
