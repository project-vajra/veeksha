from typing import List, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.config import ClientConfig, SyntheticRequestGeneratorConfig
from veeksha.core.request_config import RequestConfig
from veeksha.request_generator.synthetic_generator import SyntheticRequestGenerator
from veeksha.request_generator.length_generator.trace_generator import (
    TraceRequestLengthGenerator,
)

from veeksha.logger import init_logger

logger = init_logger(__name__)

class PrefixRequestGenerator(SyntheticRequestGenerator):

    def __init__(
        self,
        config: SyntheticRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
    ):
        super().__init__(config, tokenizer, client_config)

        assert (
            isinstance(self.request_length_generator, TraceRequestLengthGenerator)
            and self.request_length_generator.has_hash_ids()
        ), "PrefixRequestGenerator requires a TraceRequestLengthGenerator with hash IDs"
        
        self.past_prompts: dict[int, str] = {}
    
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

    def get_request(self) -> RequestConfig:
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
            id=self.request_id,
        )

        self.request_id += 1

        return request_config
