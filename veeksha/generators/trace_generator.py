from typing import List, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.config import ClientConfig, TraceRequestGeneratorConfig, TraceRequestLengthGeneratorConfig, TraceRequestIntervalGeneratorConfig
from veeksha.core.request_config import RequestConfig
from veeksha.logger import init_logger
from veeksha.generators.base_generator import BaseRequestGenerator
from veeksha.generators.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.types import RequestLengthGeneratorType, RequestIntervalGeneratorType

logger = init_logger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: TraceRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
    ):
        from veeksha.generators.generator_registry import (
            SessionGeneratorRegistry,
        )
        self.config = config
        self.tokenizer = tokenizer
        self.request_id = 0
        self.client_config = client_config
        self.past_prompts: dict[int, str] = {}

        if self.config.session_generator_config is not None:
            self.session_generator = SessionGeneratorRegistry.get(
                self.config.session_generator_config.get_type(),
                self.config.session_generator_config,
            )

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            RequestLengthGeneratorType.TRACE,
            TraceRequestLengthGeneratorConfig(
                trace_file=self.config.trace_file,
                prefill_scale_factor=self.config.prefill_scale_factor,
                decode_scale_factor=self.config.decode_scale_factor,
                block_size=self.config.block_size,
            )
        )

        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            RequestIntervalGeneratorType.TRACE,
            TraceRequestIntervalGeneratorConfig(
                trace_file=self.config.trace_file,
                time_scale_factor=self.config.time_scale_factor,
            )
        )

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
