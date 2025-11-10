from typing import Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.file_utils import load_corpus
from veeksha.config.client_config import ClientConfig
from veeksha.config.generators.request_generator.synthetic_generator_config import (
    SyntheticRequestGeneratorConfig,
)
from veeksha.core.request_config import Request
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.generators.length_generator.generator_registry import (
    RequestLengthGeneratorRegistry,
)
from veeksha.generators.request_generator.base_generator import BaseRequestGenerator
from veeksha.generators.utils import generate_random_prompt
from veeksha.logger import init_logger

logger = init_logger(__name__)


class SyntheticRequestGenerator(BaseRequestGenerator):
    def __init__(
        self,
        config: SyntheticRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        client_config: ClientConfig,
    ):
        self.config = config
        self.tokenizer = tokenizer

        self.client_config = client_config
        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
        )
        self.requests_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
        )
        self.corpus_lines = load_corpus()

        self.request_id = 0

    def get_request(self) -> Request:
        (
            num_prompt_tokens,
            num_output_tokens,
        ) = self.request_length_generator.get_next_num_tokens()
        dispatch_delay = self.requests_interval_generator.get_next_inter_request_time()
        # graceful stop if any generator signals stop via sentinel values
        if num_prompt_tokens < 0 or num_output_tokens < 0 or dispatch_delay < 0:
            return Request(
                model=self.client_config.model,
                prompt=("", 0),
                dispatch_delay=-1,
                llm_api=self.client_config.llm_api,
                id=self.request_id,
            )
        num_prompt_tokens = int(num_prompt_tokens)
        num_output_tokens = int(num_output_tokens)
        prompt_body, _ = generate_random_prompt(
            tokenizer=self.tokenizer,
            num_prompt_tokens=num_prompt_tokens,
            corpus_lines=self.corpus_lines,
        )

        instruction = f"Generate at least {num_output_tokens} tokens repeating the following text:\n"
        prompt = instruction + prompt_body

        prompt_token_count = len(self.tokenizer.encode(prompt))

        default_sampling_params = {
            "max_tokens": num_output_tokens,
        }
        default_sampling_params.update(
            self.client_config.additional_sampling_params_dict
        )
        request_config = Request(
            model=self.client_config.model,
            prompt=(prompt, prompt_token_count),
            dispatch_delay=dispatch_delay,
            sampling_params=default_sampling_params,
            llm_api=self.client_config.llm_api,
            id=self.request_id,
        )

        self.request_id += 1

        return request_config
