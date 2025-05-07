import math
import random
import os
from typing import Generator, List, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.config import SyntheticRequestGeneratorConfig
from veeksha.datatypes.request_config import RequestConfig
from veeksha.logger import init_logger
from veeksha.request_generator.base_generator import BaseRequestGenerator
from veeksha.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)

logger = init_logger(__name__)

CORPUS_RELATIVE_PATH = "../../data/corpus.txt"
CORPUS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), CORPUS_RELATIVE_PATH)
)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: SyntheticRequestGeneratorConfig,
        request_length_generator: BaseRequestLengthGenerator,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        self.config: SyntheticRequestGeneratorConfig = config
        self.request_length_generator: BaseRequestLengthGenerator = request_length_generator
        self.tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = tokenizer

        self.request_id: int = 0

    def load_corpus(self) -> List[str]:
        """Load the corpus lines from the corpus.txt file."""
        with open(CORPUS_PATH, "r") as f:
            corpus_lines = f.readlines()
        return corpus_lines

    def get_corpus_line(self) -> Generator[str, None, None]:
        # create an infinite generator of corpus lines
        # shuffle the corpus lines
        corpus_lines = self.load_corpus()
        while True:
            random.shuffle(corpus_lines)
            for line in corpus_lines:
                yield line

    def generate_random_prompt(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_prompt_tokens: int = 1024,
        num_output_tokens: int = 128,
        add_instruction: bool = True,
    ) -> Tuple[str, int]:
        """Generate a random prompt with a given number of tokens.

        Args:
            num_prompt_tokens: The number of tokens to generate in the prompt.
            num_output_tokens: The number of tokens to expect in the output.

            The prompt will be generated such that the output
            will be approximately this many tokens.

        Returns:
            A random prompt with the given number of tokens.
        """
        get_token_length = lambda text: len(tokenizer.encode(text))

        instruction = (
            'INSTRUCTION: Mimic below text enclosed in """ quotes and generate '
            f"long text of at least {num_output_tokens} tokens.\n\n"
        )

        remaining_prompt_tokens = num_prompt_tokens - get_token_length(instruction)

        sampling_lines = True
        prompt = (instruction + '"""') if add_instruction else ""
        remaining_prompt_tokens -= get_token_length(prompt) * 2
        while sampling_lines:
            for line in self.get_corpus_line():
                line_to_add = line
                if remaining_prompt_tokens - get_token_length(line_to_add) < 0:
                    # This will cut off a line in the middle of a word, but that's ok since an
                    # llm should be able to handle that.
                    line_to_add = line_to_add[: int(math.ceil(remaining_prompt_tokens))]
                    sampling_lines = False
                    prompt += line_to_add
                    break
                prompt += line_to_add
                remaining_prompt_tokens -= get_token_length(line_to_add)

        if add_instruction:
            prompt += '"""'
        return (prompt, num_prompt_tokens)

    def get_request(self) -> RequestConfig:
        (
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        if num_prompt_tokens < 0 or num_output_tokens < 0:
            logger.error(
                f"Invalid number of tokens generated: prompt={num_prompt_tokens}, output={num_output_tokens} (potentially from trace request length generator)."
            )
        num_prompt_tokens = int(num_prompt_tokens)
        num_output_tokens = int(num_output_tokens)
        prompt = self.generate_random_prompt(
            tokenizer=self.tokenizer,
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
        )
        sampling_params = {"max_tokens": num_output_tokens, "ignore_eos": True}
        request_config = RequestConfig(
            id=self.request_id,
            prompt=prompt,
            num_prompt_tokens=num_prompt_tokens,
            sampling_params=sampling_params,
        )

        self.request_id += 1

        return request_config
