from typing import Optional

from veeksha.benchmark_data_utils import load_corpus
from veeksha.config.generator.channel import TextChannelGeneratorConfig
from veeksha.core.request_content import TextChannelRequestContent
from veeksha.core.seeding import SeedManager
from veeksha.core.tokenizer import TokenizerHandle, gen_prompt_from_corpus
from veeksha.generator.channel.base import BaseChannelGenerator
from veeksha.generator.length.registry import LengthGeneratorRegistry
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TextChannelGenerator(BaseChannelGenerator):
    """Generator for text channel input content.

    This generator produces text input content for requests.
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
        corpus_lines = [line.strip() for line in load_corpus()]
        self._corpus_lines = [
            list(self.tokenizer_handle.encode(line)) for line in corpus_lines if line
        ]
        self._corpus_rng = self.seed_manager.random("text_corpus")

        self._shared_prefix_tokens: list[int] = []
        self._prefix_rng = self.seed_manager.random("shared_prefix")

    def _generate_shared_prefix(self, num_tokens: int) -> list[int]:
        """Generate and cache shared prefix tokens. Extends existing prefix if needed."""
        if len(self._shared_prefix_tokens) < num_tokens:
            # Extend the existing prefix with additional tokens
            tokens_needed = num_tokens - len(self._shared_prefix_tokens)
            additional_text = gen_prompt_from_corpus(
                num_tokens=tokens_needed,
                pretokenized_lines=self._corpus_lines,
                tokenizer_handle=self.tokenizer_handle,
                rng=self._prefix_rng,
                suffix="",
            )
            self._shared_prefix_tokens.extend(
                list(self.tokenizer_handle.encode(additional_text))
            )
        return self._shared_prefix_tokens[:num_tokens]

    def generate_content(
        self,
        is_root: bool = False,
        min_tokens_suffix: Optional[int] = None,
    ) -> TextChannelRequestContent:
        """Generate text channel content.

        Args:
            is_root: Whether this is a root request (for shared prefix handling).
            min_tokens_suffix: If provided, appends a suffix instruction requesting
                at least this many output tokens. This is used when
                use_min_tokens_prompt_fallback is enabled in the client.

        Returns:
            TextChannelRequestContent with the generated input text.
        """
        text_token_length = self.body_length_generator.get_next_value()

        use_shared_prefix = (
            is_root
            and self.config.shared_prefix_ratio > 0
            and self._corpus_rng.random() < self.config.shared_prefix_probability
        )

        # when using the shared prefix + instruction suffix, there are things to consider
        if use_shared_prefix:
            prefix_length = int(text_token_length * self.config.shared_prefix_ratio)
            remainder_length = text_token_length - prefix_length
            effective_length = remainder_length
        else:
            prefix_length = 0
            remainder_length = text_token_length
            effective_length = text_token_length

        # Build suffix for min tokens instruction if requested
        suffix = ""
        if min_tokens_suffix is not None:
            suffix = f"\n\nGenerate at least {min_tokens_suffix} tokens."
            suffix_tokens = len(self.tokenizer_handle.encode(suffix))
            if effective_length <= suffix_tokens:
                if not self._logged_body_length_warning:
                    logger.warning(
                        f"Effective body length ({effective_length}) is too short to append "
                        f"min tokens instruction ({suffix_tokens} tokens). "
                        "Skipping instruction for this request."
                    )
                    self._logged_body_length_warning = True
                suffix = ""

        if use_shared_prefix:
            prefix_tokens = self._generate_shared_prefix(prefix_length)

            # advance rng for unique remainder of each root request
            self._corpus_rng.random()

            remainder_text = gen_prompt_from_corpus(
                num_tokens=remainder_length,
                pretokenized_lines=self._corpus_lines,
                tokenizer_handle=self.tokenizer_handle,
                rng=self._corpus_rng,
                suffix=suffix,
            )

            prefix_text = self.tokenizer_handle.decode(prefix_tokens)
            input_text = prefix_text + " " + remainder_text
        else:
            input_text = gen_prompt_from_corpus(
                num_tokens=text_token_length,
                pretokenized_lines=self._corpus_lines,
                tokenizer_handle=self.tokenizer_handle,
                rng=self._corpus_rng,
                suffix=suffix,
            )

        return TextChannelRequestContent(
            input_text=input_text,
            target_prompt_tokens=text_token_length,
        )
