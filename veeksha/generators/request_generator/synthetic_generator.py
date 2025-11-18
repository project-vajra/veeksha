from typing import Any, Dict, List, Optional, Tuple, Union, cast

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.client import ClientConfig
from veeksha.config.generators.request_generator.synthetic_generator import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.core.request_config import RequestConfig
from veeksha.core.seeding import SeedManager
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.generators.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import generate_random_token_ids_fast
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: SyntheticRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
        seed_manager: SeedManager,
        corpus_lines: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.seed_manager = seed_manager

        self.client_config = client_config
        sm = self.seed_manager
        self.interval_rng_factory = sm.numpy_factory("interval")
        self.length_rng_factory = sm.numpy_factory("length")
        self.prompt_rng = sm.random("prompt")

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
            rng=self.length_rng_factory(),
        )
        self.requests_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
            rng=self.interval_rng_factory(),
        )
        self.corpus_lines = corpus_lines

        # pre-tokenize
        logger.info("Pre-tokenizing corpus.")
        self.pretokenized_lines: List[List[int]] = []
        if corpus_lines is not None:
            token_lines = [
                self.tokenizer.encode(line, add_special_tokens=False)
                for line in corpus_lines
            ]
            self.pretokenized_lines = [t for t in token_lines if t]

        # cache instructions from 10 to 1000 tokens
        self._instruction_cache: Dict[int, List[int]] = {}
        for n in range(10, 1001):
            instr_text = f"Generate at least {n} tokens repeating the following text:\n"
            self._instruction_cache[n] = self.tokenizer.encode(
                instr_text, add_special_tokens=False
            )

        self._global_request_id = 0

    def _get_instruction_ids(
        self, num_output_tokens: int, use_server_min_tokens: bool
    ) -> List[int]:
        """Return cached instruction token IDs or empty if server min-tokens is supported."""
        if use_server_min_tokens:
            return []
        instr_ids = self._instruction_cache.get(num_output_tokens)
        if instr_ids is not None:
            return instr_ids
        instr_text = f"Generate at least {num_output_tokens} tokens repeating the following text:\n"
        instr_ids = self.tokenizer.encode(instr_text, add_special_tokens=False)
        self._instruction_cache[num_output_tokens] = instr_ids
        return instr_ids

    def _generate_body_ids(self, body_token_count: int) -> List[int]:
        """Generate exactly body_token_count token IDs from the pre-tokenized corpus."""
        if body_token_count <= 0:
            return []
        return generate_random_token_ids_fast(
            pretokenized_lines=self.pretokenized_lines,
            num_tokens=body_token_count,
            rng=self.prompt_rng,
        )

    def _assemble_prompt(
        self,
        num_prompt_tokens: int,
        num_output_tokens: int,
        use_server_min_tokens: bool,
    ) -> Tuple[str, int]:
        """Build prompt text and exact token count using cached instruction and fast body IDs."""
        instr_ids = self._get_instruction_ids(num_output_tokens, use_server_min_tokens)
        body_token_count = max(0, int(num_prompt_tokens) - len(instr_ids))
        body_ids = self._generate_body_ids(body_token_count)
        full_ids = instr_ids + body_ids
        prompt = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        return prompt, len(full_ids)

    def get_request(self) -> RequestConfig:
        (
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        dispatch_delay = self.requests_interval_generator.get_next_inter_request_time()
        # graceful stop if any generator signals stop via sentinel values
        if num_prompt_tokens < 0 or num_output_tokens < 0 or dispatch_delay < 0:
            return RequestConfig(
                model=self.client_config.model,
                prompt=("", 0),
                dispatch_delay=-1,
                llm_api=self.client_config.llm_api,
                address_append_value=self.client_config.address_append_value,
                id=self._global_request_id,
            )
        num_prompt_tokens = int(num_prompt_tokens)
        num_output_tokens = int(num_output_tokens)

        # Use server-side min_tokens if available (probing already validated support)
        use_server_min_tokens = self.client_config.min_tokens_param is not None
        prompt, prompt_token_count = self._assemble_prompt(
            num_prompt_tokens=num_prompt_tokens,
            num_output_tokens=num_output_tokens,
            use_server_min_tokens=use_server_min_tokens,
        )

        default_sampling_params: Dict[str, Any] = {
            "max_completion_tokens": num_output_tokens,
        }
        if use_server_min_tokens:
            min_token_value = num_output_tokens
            min_tokens_param_name = cast(str, self.client_config.min_tokens_param)
            default_sampling_params[min_tokens_param_name] = min_token_value
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )

        request_config = RequestConfig(
            model=self.client_config.model,
            prompt=(prompt, prompt_token_count),
            dispatch_delay=dispatch_delay,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            address_append_value=self.client_config.address_append_value,
            id=self._global_request_id,
        )

        self._global_request_id += 1
        return request_config
