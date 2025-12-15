from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, List, Sequence, TypeVar

from transformers import AutoTokenizer

from veeksha.new.types import ChannelModality

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


class TokenizerProvider:
    """Lightweight provider that returns a tokenizer handle per modality."""

    def __init__(self, tokenizers: Dict[ChannelModality, TokenizerHandle[Any]]):
        self._tokenizers = tokenizers

    def for_modality(self, modality: ChannelModality) -> TokenizerHandle[Any]:
        return self._tokenizers[modality]


def build_hf_tokenizer_handle(tokenizer) -> TokenizerHandle[str]:
    """Wrap a Hugging Face tokenizer into a TokenizerHandle."""

    return TokenizerHandle(
        count_tokens=lambda text: len(tokenizer.encode(text)),
        decode=lambda token_ids: tokenizer.decode(token_ids, skip_special_tokens=False),
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
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
) -> RawContent:
    """Assemble exactly num_tokens token IDs from a pre-tokenized corpus."""

    token_lines: List[Sequence[int]] = [line for line in pretokenized_lines if line]
    if num_tokens <= 0:
        return tokenizer_handle.decode([])
    remaining = num_tokens
    out: List[int] = []
    indices = list(range(len(token_lines)))
    rng.shuffle(indices)
    idx_cursor = 0
    while remaining > 0:
        tokens = token_lines[indices[idx_cursor]]
        take = min(remaining, len(tokens))
        if take:
            out.extend(tokens[:take])
            remaining -= take
        idx_cursor += 1
        if idx_cursor == len(indices):
            idx_cursor = 0
            rng.shuffle(indices)
    return tokenizer_handle.decode(out)
