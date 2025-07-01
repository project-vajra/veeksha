from typing import Dict, List, Union
import ast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.generators.request_generator.trace_generator import TraceRequestGeneratorConfig
from veeksha.config.client import ClientConfig
from veeksha.core.request_config import RequestConfig
from veeksha.logger import init_logger
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import process_request_length_trace, process_request_interval_trace, load_trace

logger = init_logger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: TraceRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
    ):
        from veeksha.generators.session_generator.generator_registry import (
            SessionGeneratorRegistry,
        )

        self.config = config
        self.tokenizer = tokenizer
        self.request_id = 0
        self.client_config = client_config
        self.past_prompts: Dict[int, str] = {}

        self.trace_df = load_trace(self.config.trace_file)

        self.trace_df = process_request_length_trace(
            self.trace_df,
            self.config.trace_file,
            self.config.prefill_scale_factor,
            self.config.decode_scale_factor,
            self.config.max_tokens,
        )

        self.trace_df = process_request_interval_trace(
            self.trace_df,
            self.config.trace_file,
            self.config.time_scale_factor,
        )

        logger.info(
            f"Loaded trace file {self.config.trace_file} with {len(self.trace_df)} requests"
        )

        self._has_hash_ids = "hash_ids" in self.trace_df.columns

        # parse, or not, hash_ids
        if self.config.use_trace_prefix_hash_ids:
            if not self._has_hash_ids:
                raise ValueError("Trace file does not contain hash_ids")
            else:
                if not isinstance(self.trace_df["hash_ids"].iloc[0], list):
                    self.trace_df["hash_ids"] = self.trace_df["hash_ids"].apply(
                        ast.literal_eval
                    )

        if self.config.use_trace_sessions:
            # TODO: implement
            raise NotImplementedError("to be implemented")
        elif self.config.session_generator_config is not None:
            self.session_generator = SessionGeneratorRegistry.get(
                self.config.session_generator_config.get_type(),
                self.config.session_generator_config,
            )
            self.trace_df = self.session_generator.generate_sessions(self.trace_df)

            # get next request intervals again, because session sampling might shuffle requests
            self.trace_df = process_request_interval_trace(
                self.trace_df,
                self.config.trace_file,
                self.config.time_scale_factor,
                ms_to_s=False,
            )

            self.session_generator.save_requests_as_trace(self.trace_df)

        self.request_idx = 0

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
        request_to_send = self.trace_df.iloc[self.request_idx]

        request_metadata = {
            "input_length": request_to_send["input_length"],
            "output_length": request_to_send["output_length"],
            "request_dispatch_interval": request_to_send["inter_request_time"],
        }

        if self.config.use_trace_sessions or self.config.session_generator_config is not None:
            request_metadata["session_id"] = request_to_send["session_id"]
            request_metadata["session_size"] = request_to_send["num_requests_in_session"]

        if self.config.use_trace_prefix_hash_ids:
            block_count = (
                request_to_send["input_length"] + self.config.block_size - 1
            ) // self.config.block_size

            request_metadata["block_count"] = block_count

            assert len(request_to_send["hash_ids"]) >= block_count, f"Hash count {len(request_to_send['hash_ids'])} cannot be less than block count {block_count}"

        prompt = ""
        remaining_prompt_tokens = request_to_send["input_length"]
        if self.config.use_trace_prefix_hash_ids:
            for hash_id in request_to_send["hash_ids"]:
                if hash_id not in self.past_prompts:
                    chunk = self.generate_unique_encoding(hash_id)
                    block = self.pad_to_block_size(chunk, self.config.block_size)
                    prompt_segment = self.decode(block)
                    remaining_prompt_tokens -= self.config.block_size
                    self.past_prompts[hash_id] = prompt_segment
                prompt += self.past_prompts[hash_id]
        else:
            # todo input text
            raise NotImplementedError("to be implemented")

        final_token_count = len(self.encode(prompt))

        default_sampling_params = {"min_tokens": int(request_to_send['output_length']), "max_tokens": int(request_to_send['output_length'])}
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, final_token_count),
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=self.request_idx,
            metadata=request_metadata,
        )

        self.request_idx += 1

        return request_config