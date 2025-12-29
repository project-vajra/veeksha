from veeksha.benchmark_data_utils import load_corpus
from veeksha.logger import init_logger
from veeksha.new.config.generator.channel import TextChannelGeneratorConfig
from veeksha.new.core.request_content import TextChannelRequestContent
from veeksha.new.core.seeding import SeedManager
from veeksha.new.core.tokenizer import TokenizerHandle, gen_prompt_from_corpus
from veeksha.new.generator.channel.base import BaseChannelGenerator
from veeksha.new.generator.length.registry import LengthGeneratorRegistry

logger = init_logger(__name__)


class TextChannelGenerator(BaseChannelGenerator):
    def __init__(
        self,
        config: TextChannelGeneratorConfig,
        seed_manager: SeedManager,
        tokenizer_handle: TokenizerHandle,
        append_min_tokens_instruction: bool = False,
    ):
        self.config = config
        self._logged_body_length_warning = False
        self.seed_manager = seed_manager
        self.append_min_tokens_instruction = append_min_tokens_instruction
        self.body_length_generator = LengthGeneratorRegistry.get(
            self.config.body_length_generator.get_type(),
            self.config.body_length_generator,
            rng=self.seed_manager.numpy_factory("body_length")(),
        )
        self.output_length_generator = LengthGeneratorRegistry.get(
            self.config.output_length_generator.get_type(),
            self.config.output_length_generator,
            rng=self.seed_manager.numpy_factory("output_length")(),
        )
        self.tokenizer_handle = tokenizer_handle
        # TODO: load corpus function to .new
        corpus_lines = [line.strip() for line in load_corpus()]
        self._corpus_lines = [
            list(self.tokenizer_handle.encode(line)) for line in corpus_lines if line
        ]
        self._corpus_rng = self.seed_manager.random("text_corpus")

    def generate_content(self) -> TextChannelRequestContent:
        text_token_length = self.body_length_generator.get_next_value()
        output_token_length = self.output_length_generator.get_next_value()

        suffix = ""
        if self.append_min_tokens_instruction:
            suffix = f"\n\nGenerate at least {output_token_length} tokens."
            suffix_tokens = len(self.tokenizer_handle.encode(suffix))
            if text_token_length <= suffix_tokens:
                if not self._logged_body_length_warning:
                    logger.warning(
                        f"Requested body length ({text_token_length}) is too short to append "
                        f"min tokens instruction ({suffix_tokens} tokens). "
                        "Skipping instruction for this request."
                    )
                    self._logged_body_length_warning = True
                suffix = ""

        input_text = gen_prompt_from_corpus(
            num_tokens=text_token_length,
            pretokenized_lines=self._corpus_lines,
            tokenizer_handle=self.tokenizer_handle,
            rng=self._corpus_rng,
            suffix=suffix,
        )
        return TextChannelRequestContent(
            input_text=input_text,
            target_output_tokens=output_token_length,
            target_prompt_tokens=text_token_length,
        )
