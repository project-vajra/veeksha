from typing import List, Optional, Union

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from veeksha.config.generators.request_generator.base_generator_config import (
    BaseRequestGeneratorConfig,
)
from veeksha.core.request_config import Request
from veeksha.core.response import Response
from veeksha.logger import init_logger

logger = init_logger(__name__)


class BaseRequestGenerator:
    def __init__(
        self,
        config: BaseRequestGeneratorConfig,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ):
        self.config = config
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def get_request(self) -> Request:
        raise NotImplementedError

    def get_responses(self, responses: List[Response]) -> None:
        pass

    def capacity(self) -> Optional[int]:
        """Optional: total number of requests producible if finite; None if unbounded."""
        return None
