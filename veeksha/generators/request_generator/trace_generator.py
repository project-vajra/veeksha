import ast
from typing import Any, Dict, List, Optional, Union, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.trace_generator import (
    TraceRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.seeding import SeedManager
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import (
    generate_random_prompt,
    load_trace,
    process_request_interval_trace,
    process_request_length_trace,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TraceRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: TraceRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
        seed_manager: SeedManager,
        corpus_lines: Optional[List[str]] = None,
    ):
        from veeksha.generators.session_generator import (
            SessionGenerator,
        )

        self.config = config
        self.tokenizer = tokenizer
        self.seed_manager = seed_manager
        self.request_id = 0
        self.client_config = client_config
        self.past_prompts: Dict[int, str] = {}
        self.corpus_lines = corpus_lines
        self._remap_seed_for_save: Optional[int] = None
        sm = self.seed_manager
        self.prompt_rng = sm.random("prompt")
        self.interval_rng_factory = sm.numpy_factory("interval")
        self.session_rng_factory = sm.numpy_factory("session")
        self.rng = self.prompt_rng

        raw_trace_df = load_trace(self.config.trace_file)

        # canonical column names
        self.length_column_map = {
            self.config.input_length_column: "input_length",
            self.config.output_length_column: "output_length",
        }
        self.interval_column_map = {self.config.timestamp_column: "timestamp"}

        self.trace_df = raw_trace_df.pipe(
            process_request_length_trace,
            self.config.trace_file,
            self.length_column_map,
            self.config.prefill_scale_factor,
            self.config.decode_scale_factor,
            self.config.max_tokens,
        ).pipe(
            process_request_interval_trace,
            self.config.trace_file,
            self.interval_column_map,
            self.config.time_scale_factor,
            self.config.timestamp_unit,
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
                if self.config.remap_hash_ids:
                    unique_ids = set()
                    for ids in self.trace_df["hash_ids"]:
                        unique_ids.update(ids)
                    # unbias permutation
                    unique_list = sorted(unique_ids)
                    permuted = unique_list.copy()
                    rng = self.session_rng_factory()
                    rng.shuffle(permuted)
                    id_map: Dict[int, int] = {
                        src: dst for src, dst in zip(unique_list, permuted)
                    }
                    logger.info(
                        f"Applying hash-id remapping with session RNG to {len(unique_list)} unique ids"
                    )
                    self.trace_df["hash_ids"] = self.trace_df["hash_ids"].apply(
                        lambda lst: [id_map[x] for x in lst]
                    )
        else:
            if self.corpus_lines is None:
                raise ValueError(
                    "A corpus file must be provided when not using trace prefix hash IDs."
                )

        if self.config.use_trace_sessions:
            if "session_id" not in self.trace_df.columns:
                raise ValueError("Trace file does not contain session_id of requests")
        elif self.config.session_generator_config is not None:
            session_generator = SessionGenerator(
                self.config.session_generator_config,
                seed_manager=self.seed_manager.child("session"),
            )

            self.trace_df_with_sessions = self.trace_df.pipe(
                session_generator.generate_sessions,
            ).pipe(
                # get next request intervals again because session sampling shuffles sessions
                process_request_interval_trace,
                self.config.trace_file,
                None,  # colnames are already canonical
                self.config.time_scale_factor,
                "s",  # self.trace_df has already been converted to seconds
            )

            if self.config.session_generator_config.save_as_trace_file:
                # convert timestamps to milliseconds (default time units) before saving
                session_df_for_saving = self.trace_df_with_sessions.copy()
                session_df_for_saving["timestamp"] = (
                    session_df_for_saving["timestamp"] * 1000
                )
                save_suffix = f"_remapped" if self.config.remap_hash_ids else ""
                session_generator.save_requests_as_trace(
                    session_df_for_saving,
                    save_suffix=save_suffix,
                )

        self.request_idx = 0
        self._wrap_warning_logged = False

    def is_stable_encoding(
        self,
        tokens: List[int],
    ) -> bool:
        return self.encode(self.decode(tokens)) == tokens

    def encode_value_as_base_52(self, value: int) -> List[int]:
        if value <= 0:
            raise ValueError(
                f"Value must be a positive integer for base-52 encoding, got: {value}"
            )

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
        if value <= 0:
            raise ValueError(
                f"Value must be a positive integer for digits encoding, got: {value}"
            )

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
        if self.request_idx >= self.capacity():
            if self.config.exhaustion_policy == "error":
                raise StopIteration(
                    f"Trace exhausted for requests at index {self.request_idx}"
                )
            elif self.config.exhaustion_policy == "stop":
                # stop policy: return a sentinel request with negative dispatch delay
                logger.info(
                    f"Stop policy active: request trace exhausted at index {self.request_idx}."
                )
                return RequestConfig(
                    model=self.client_config.model,
                    prompt=("", 0),
                    dispatch_delay=-1,
                    llm_api=self.client_config.llm_api,
                    address_append_value=self.client_config.address_append_value,
                    id=self.request_idx,
                )
            elif self.config.exhaustion_policy == "wrap":
                if not self._wrap_warning_logged:
                    logger.warning(
                        f"Request trace exhausted at index {self.request_idx}; wrapping to start."
                    )
                    self._wrap_warning_logged = True
                self.request_idx = 0

        if self.config.session_generator_config is not None:
            request_to_send = self.trace_df_with_sessions.iloc[self.request_idx]
        else:
            request_to_send = self.trace_df.iloc[self.request_idx]

        dispatch_delay = request_to_send["inter_request_time"]

        if self.config.use_trace_prefix_hash_ids:
            block_count = (
                request_to_send["input_length"] + self.config.block_size - 1
            ) // self.config.block_size

            assert (
                len(request_to_send["hash_ids"]) >= block_count
            ), f"Hash count {len(request_to_send['hash_ids'])} cannot be less than block count {block_count}"

        prompt = ""
        remaining_prompt_tokens = request_to_send["input_length"]
        use_server_min_tokens = self.client_config.min_tokens_param is not None
        instruction = ""
        if not use_server_min_tokens:
            instruction = f"Generate at least {int(request_to_send['output_length'])} tokens repeating the following text:\n"
            instruction_token_count = len(self.tokenizer.encode(instruction))
            remaining_prompt_tokens = max(
                0, remaining_prompt_tokens - instruction_token_count
            )

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
            # generate input random text
            prompt, _ = generate_random_prompt(
                tokenizer=self.tokenizer,
                num_prompt_tokens=remaining_prompt_tokens,
                corpus_lines=self.corpus_lines,
                rng=self.rng,
            )

        prompt = (instruction + prompt) if instruction else prompt
        final_token_count = len(self.encode(prompt))

        default_sampling_params: Dict[str, Any] = {
            "max_tokens": int(request_to_send["output_length"]),
        }
        if use_server_min_tokens:
            min_token_value = int(request_to_send["output_length"])
            min_tokens_param_name = cast(str, self.client_config.min_tokens_param)
            default_sampling_params[min_tokens_param_name] = min_token_value
        # else prompt already includes instruction
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, final_token_count),
            dispatch_delay=dispatch_delay,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=self.request_idx,
        )

        # attach session scheduling metadata when enabled
        if self.config.session_generator_config is not None:
            session_policy = (
                self.config.session_generator_config.session_dispatch_policy
            )
            cancel_on_failure = (
                self.config.session_generator_config.cancel_session_on_failure
            )

            request_config.session_id = int(request_to_send.get("session_id"))
            seq_idx = int(request_to_send.get("session_sequence_index", 0))
            request_config.session_sequence_index = seq_idx
            request_config.cancel_session_on_failure = bool(cancel_on_failure)

            if session_policy == "after_prev_response":
                # Only first-in-session gets anchor; others use wait_after_prev_response
                anchor = float(request_to_send.get("anchor_at_s", 0.0))
                if seq_idx == 0 and anchor > 0.0:
                    request_config.anchor_at_s = anchor
                wait_gap = float(request_to_send.get("wait_after_prev_response_s", 0.0))
                if seq_idx > 0:
                    request_config.wait_after_prev_response_s = wait_gap

        self.request_idx += 1

        return request_config

    def capacity(self) -> int:
        return (
            len(self.trace_df)
            if self.config.session_generator_config is None
            else len(self.trace_df_with_sessions)
        )
