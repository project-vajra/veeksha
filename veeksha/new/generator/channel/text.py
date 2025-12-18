from veeksha.benchmark_data_utils import load_corpus
from veeksha.new.config.generator.channel import TextChannelGeneratorConfig
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.tokenizer import TokenizerHandle, gen_prompt_from_corpus
from veeksha.new.generator.channel.base import BaseChannelGenerator
from veeksha.new.generator.length.registry import LengthGeneratorRegistry


class TextChannelGenerator(BaseChannelGenerator):
    def __init__(
        self,
        config: TextChannelGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_handle: TokenizerHandle,
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.length_generator = LengthGeneratorRegistry.get(
            self.config.length_generator.get_type(),
            self.config.length_generator,
            rng=self.seed_manager.numpy_factory("length")(),
        )
        self.tokenizer_handle = tokenizer_handle
        # TODO: load corpus function to .new
        corpus_lines = [line.strip() for line in load_corpus()]
        self._corpus_lines = [
            list(self.tokenizer_handle.encode(line)) for line in corpus_lines if line
        ]
        self._corpus_rng = self.seed_manager.random("text_corpus")

    def generate_content(self) -> str:
        text_token_length = self.length_generator.get_next_length()
        return gen_prompt_from_corpus(
            num_tokens=text_token_length,
            pretokenized_lines=self._corpus_lines,
            tokenizer_handle=self.tokenizer_handle,
            rng=self._corpus_rng,
        )
