from typing import Optional

from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.core.prompt_generator import PromptStringGenerator
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.core.tokenizer import TokenizerHandle
from veeksha.generator.channel.base import BaseChannelGenerator
from veeksha.generator.length.registry import LengthGeneratorRegistry
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TextChannelGenerator(BaseChannelGenerator):
    """Generator for text channel input content.

    Uses PromptStringGenerator: stable encodings are loaded from disk cache
    (~10 ms) or generated once (~2 s) and then pre-decoded.  Per-session
    cost is pure Python string ops (~0.5 ms), with no tokenizer calls.
    """

    def __init__(
        self,
        config: TextChannelGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_handle: TokenizerHandle,
    ):
        self.config = config
        self._logged_body_length_warning = False
        self.seed_manager = seed_manager
        self.body_length_generator = LengthGeneratorRegistry.get(
            self.config.body_length_generator.get_type(),
            self.config.body_length_generator,
            rng=self.seed_manager.numpy_factory("body_length")(),
        )
        self.tokenizer_handle = tokenizer_handle
        self._prompt_gen = PromptStringGenerator(
            tokenizer_handle,
            rng=self.seed_manager.random("prompt_gen"),
        )
        self._shared_prefix_tokens: list[int] = []
        self._prefix_rng = self.seed_manager.random("shared_prefix")

    def _generate_shared_prefix(self, num_tokens: int) -> list[int]:
        """Generate and cache shared prefix tokens. Extends if needed."""
        if len(self._shared_prefix_tokens) < num_tokens:
            tokens_needed = num_tokens - len(self._shared_prefix_tokens)
            additional_text = self._prompt_gen.generate(tokens_needed)
            self._shared_prefix_tokens.extend(
                list(self.tokenizer_handle.encode(additional_text))
            )
        return self._shared_prefix_tokens[:num_tokens]

    def generate_content(
        self,
        is_root: bool = False,
        min_tokens_suffix: Optional[int] = None,
    ) -> TextChannelRequestContent:
        """Generate text channel content."""
        text_token_length = self.body_length_generator.get_next_value()

        use_shared_prefix = (
            is_root
            and self.config.shared_prefix_ratio > 0
            and self._prefix_rng.random() < self.config.shared_prefix_probability
        )

        if use_shared_prefix:
            prefix_length = int(text_token_length * self.config.shared_prefix_ratio)
            remainder_length = text_token_length - prefix_length
            effective_length = remainder_length
        else:
            effective_length = text_token_length

        # Build suffix for min tokens instruction if requested
        suffix = ""
        if min_tokens_suffix is not None:
            suffix = f"\n\nGenerate at least {min_tokens_suffix} tokens."
            suffix_tokens = len(self.tokenizer_handle.encode(suffix))
            if effective_length <= suffix_tokens:
                if not self._logged_body_length_warning:
                    logger.warning(
                        f"Effective body length ({effective_length}) is too short to "
                        f"append min tokens instruction ({suffix_tokens} tokens). "
                        "Skipping instruction for this request."
                    )
                    self._logged_body_length_warning = True
                suffix = ""

        if use_shared_prefix:
            plen = int(text_token_length * self.config.shared_prefix_ratio)
            rlen = text_token_length - plen
            prefix_tokens = self._generate_shared_prefix(plen)
            prefix_text = self.tokenizer_handle.decode(prefix_tokens)
            remainder_text = self._prompt_gen.generate(rlen) + suffix
            input_text = prefix_text + " " + remainder_text
        else:
            input_text = self._prompt_gen.generate(text_token_length) + suffix

        return TextChannelRequestContent(
            input_text=input_text,
            target_prompt_tokens=text_token_length,
        )
