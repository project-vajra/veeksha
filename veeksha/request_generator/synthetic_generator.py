import math
import random
from typing import List, Optional, Tuple, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.config import ClientConfig
from veeksha.core.request_config import RequestConfig
from veeksha.request_generator.length_generator.base_generator import (
    BaseRequestLengthGenerator,
)
from veeksha.request_generator.length_generator.trace_generator import (
    TraceRequestLengthGenerator,
)

from veeksha.logger import init_logger

logger = init_logger(__name__)

# TODO: split this class into two classes, one for normal requests and one for prefix requests,
# and have them both inherit from a common base class.
class SyntheticRequestGenerator:

    def __init__(
        self,
        client_config: ClientConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        request_length_generator: BaseRequestLengthGenerator,
        corpus_lines: Optional[List[str]] = None,
    ):
        self.client_config = client_config
        self.tokenizer = tokenizer
        self.request_length_generator = request_length_generator
        self.corpus_lines = corpus_lines

        self.past_prompts: dict[int, str] = {}

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    @property
    def uses_prefix(self) -> bool:
        return (
            isinstance(self.request_length_generator, TraceRequestLengthGenerator)
            and self.request_length_generator.has_hash_ids()
        )

    def get_request_params(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:
        if self.uses_prefix:
            return self.get_request_params_prefix(request_id=request_id)
        else:
            return self.get_request_params_normal(request_id=request_id)
    
    def generate_random_prompt(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        num_prompt_tokens: int = 1024,
        num_output_tokens: int = 128,
        corpus_lines: Union[List[str], None] = None,
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
        assert corpus_lines is not None, "corpus_lines must be provided"

        get_token_length = lambda text: len(tokenizer.encode(text))

        instruction = (
            'INSTRUCTION: Mimic below text enclosed in """ quotes and generate '
            f"long text of at least {num_output_tokens} tokens.\n\n"
        )

        remaining_prompt_tokens = num_prompt_tokens - get_token_length(instruction)
        random.shuffle(corpus_lines)
        sampling_lines = True
        prompt = (instruction + '"""') if add_instruction else ""
        remaining_prompt_tokens -= get_token_length(prompt) * 2
        while sampling_lines:
            for line in corpus_lines:
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

    def get_request_params_normal(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:

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
            corpus_lines=self.corpus_lines,
        )
        default_sampling_params = {"max_tokens": num_output_tokens}
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )
        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=prompt,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=request_id,
        )

        return request_config

    def is_stable_encoding(
        self,
        tokens: List[int],
    ) -> bool:
        return self.encode(self.decode(tokens)) == tokens

    def encode_value_as_base_52(self, value: int) -> List[int]:
        base_52 = []
        while value > 0:
            mod = value % 52
            if mod < 26:
                base_52.append(chr(ord("a") + mod))
            else:
                base_52.append(chr(ord("A") + mod - 26))
            value = value // 52
        text_chunk = " " + "".join(base_52)
        encoding = self.encode(text_chunk)

        return encoding

    def encode_value_as_digits(self, value: int) -> List[int]:
        digits = list(str(value))
        space_separated = " " + " ".join(digits)
        encoding = self.encode(space_separated)
        return encoding

    def pad_to_block_size(self, chunk: List[int], block_size: int) -> List[int]:
        final_chunk = chunk * (block_size // len(chunk) + 1)
        return final_chunk[:block_size]

    def generate_unique_encoding(self, value: int) -> List[int]:
        encoding = self.encode_value_as_base_52(value)
        if self.is_stable_encoding(encoding + encoding):
            return encoding

        encoding = self.encode_value_as_digits(value)
        if self.is_stable_encoding(encoding + encoding):
            return encoding

        raise Exception(f"Could not generate stable encoding for value {value}")

    def get_request_params_prefix(
        self,
        request_id: Optional[int] = None,
    ) -> RequestConfig:
        assert (
            isinstance(self.request_length_generator, TraceRequestLengthGenerator)
            and self.request_length_generator.has_hash_ids()
        )

        (
            hash_ids,
            remaining_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_request_params()
        block_size = self.request_length_generator.get_block_size()

        prompt = '"""'
        for hash_id in hash_ids:
            if hash_id not in self.past_prompts:
                chunk = self.generate_unique_encoding(hash_id)
                block = self.pad_to_block_size(chunk, block_size)
                prompt_segment = self.decode(block)
                remaining_prompt_tokens -= block_size
                self.past_prompts[hash_id] = prompt_segment
            prompt += self.past_prompts[hash_id]

        prompt += '"""'

        prompt += (
            '\n\nINSTRUCTION: Mimic above text enclosed in """ quotes and generate '
            f"long text of at least {num_output_tokens} tokens."
        )

        final_token_count = len(self.encode(prompt))

        default_sampling_params = {"max_tokens": num_output_tokens}
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, final_token_count),
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=request_id,
        )

        return request_config
