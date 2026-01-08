from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
)

from transformers import AutoTokenizer

from veeksha.types import ChannelModality

RawContent = TypeVar("RawContent")
TokenIds = Sequence[int]

TokenCounter = Callable[[RawContent], int]
TokenDecoder = Callable[[TokenIds], RawContent]
TokenEncoder = Callable[[RawContent], TokenIds]


@dataclass
class TokenizerHandle(Generic[RawContent]):
    """Minimal tokenizer abstraction used by channel generators."""

    count_tokens: TokenCounter
    decode: TokenDecoder
    encode: TokenEncoder
    get_vocab: Optional[Callable[[], List[int]]] = None


class TokenizerProvider:
    """Lightweight provider that returns a tokenizer handle per modality."""

    def __init__(
        self,
        tokenizers: Dict[ChannelModality, TokenizerHandle[Any]],
        model_name: Optional[str] = None,
    ):
        self._tokenizers = tokenizers
        self._model_name = model_name

    def for_modality(self, modality: ChannelModality) -> TokenizerHandle[Any]:
        return self._tokenizers[modality]

    @property
    def model_name(self) -> Optional[str]:
        """Return the model name for loading raw tokenizers."""
        return self._model_name


def build_hf_tokenizer_handle(tokenizer) -> TokenizerHandle[str]:
    """Wrap a Hugging Face tokenizer into a TokenizerHandle."""

    # cache vocab
    vocab = sorted(tokenizer.vocab.values())[: tokenizer.vocab_size]

    return TokenizerHandle(
        count_tokens=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        decode=lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False),
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        get_vocab=lambda: vocab,
    )


def build_hf_tokenizer_handle_from_model(model: str) -> TokenizerHandle[str]:
    """Instantiate a Hugging Face tokenizer from a model name and wrap it."""

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    return build_hf_tokenizer_handle(tokenizer)


def gen_prompt_from_corpus(
    num_tokens: int,
    pretokenized_lines: Iterable[Sequence[int]],
    tokenizer_handle: TokenizerHandle[RawContent],
    rng,
    suffix: Optional[RawContent] = None,
) -> RawContent:
    """Assemble exactly num_tokens token IDs from a pre-tokenized corpus."""
    empty_content = tokenizer_handle.decode([])
    effective_suffix = suffix if suffix is not None else empty_content

    if num_tokens <= 0:
        return empty_content

    suffix_len = tokenizer_handle.count_tokens(effective_suffix)
    target_body_len = max(0, num_tokens - suffix_len)

    # 1. get candidate tokens
    token_lines = [line for line in pretokenized_lines if line]
    if not token_lines:
        return effective_suffix

    rng.shuffle(token_lines)
    candidate_ids = [tok for line in token_lines for tok in line]
    needed = int(target_body_len * 1.2) + 50
    candidate_ids = candidate_ids[: needed + 100]

    # 2. binary search for best token count
    low, high, best_k = 0, len(candidate_ids), len(candidate_ids)
    while low <= high:
        mid = (low + high) // 2
        count = tokenizer_handle.count_tokens(
            tokenizer_handle.decode(candidate_ids[:mid]) + effective_suffix
        )
        if count == num_tokens:
            return tokenizer_handle.decode(candidate_ids[:mid]) + effective_suffix
        elif count < num_tokens:
            low = mid + 1
        else:
            best_k = mid
            high = mid - 1

    # 3. trim or pad characters
    def try_adjust(base_ids: List[int], trim: bool) -> Optional[RawContent]:
        text = tokenizer_handle.decode(base_ids)
        limit = 200 if trim else 50
        for i in range(limit):
            adjusted = text[:-i] if trim and i > 0 else text + (" " * i)
            if trim and i >= len(text):
                break
            if tokenizer_handle.count_tokens(adjusted + effective_suffix) == num_tokens:
                return adjusted + effective_suffix
        return None

    # overshoot
    if result := try_adjust(candidate_ids[:best_k], trim=True):
        return result
    # undershoot
    if best_k > 0 and (result := try_adjust(candidate_ids[: best_k - 1], trim=False)):
        return result

    return tokenizer_handle.decode(candidate_ids[:best_k]) + effective_suffix
