"""Prompt builder for trace-driven session generators.

Ported from xyz's SeshChef - provides utilities for generating prompts
from trace data with corpus sampling and hash ID-based content.
"""

import random
from pathlib import Path
from typing import List, Optional

from veeksha.core.seeding import SeedManager
from veeksha.core.tokenizer import gen_prompt_from_corpus


def base10_to_basen(x: int, n: int) -> List[int]:
    """Convert a base-10 integer to base-n representation."""
    assert x >= 0
    assert n >= 2
    digits = []
    while x > 0:
        digits.append(x % n)
        x = x // n
    digits.reverse()
    return digits


class TracePromptBuilder:
    """Builds prompts from trace data with optional corpus sampling.

    This utility class provides methods for:
    - Generating unique deterministic prompts from seeds
    - Sampling tokens from a corpus file
    - Building prompts from hash IDs (for MooncakeConv traces)
    """

    def __init__(
        self,
        tokenizer,
        seed_manager: SeedManager,
        corpus_file: Optional[Path] = None,
    ):
        """Initialize the prompt builder.

        Args:
            tokenizer: TokenizerHandle for encoding/decoding.
            seed_manager: Seed manager for random state.
            corpus_file: Optional path to corpus file for sampling.
        """
        self.tokenizer = tokenizer
        self.rng = seed_manager.random("prompt_builder_sampler")

        # pre-cache sorted vocab for generate_unique_prompt
        if tokenizer.get_vocab is None:
            raise ValueError(
                "Tokenizer handle must support get_vocab for TracePromptBuilder"
            )
        self._sorted_vocab = tokenizer.get_vocab()
        self._sorted_vocab = tokenizer.get_vocab()

        # Pre-tokenize corpus if provided
        self.pretokenized_lines: List[List[int]] = []
        if corpus_file is not None:
            corpus_path = Path(corpus_file)
            if corpus_path.exists():
                with open(corpus_path, "r", encoding="utf-8") as f:
                    corpus_lines = f.readlines()
                self.pretokenized_lines = [
                    self.tokenizer.encode(line) for line in corpus_lines
                ]

                total_tokens = sum(len(line) for line in self.pretokenized_lines)
                min_tokens_needed = 200_000

                if total_tokens > 0 and total_tokens < min_tokens_needed:
                    multiplier = (min_tokens_needed // total_tokens) + 1
                    self.pretokenized_lines = self.pretokenized_lines * multiplier

    def generate_unique_prompt(self, num_tokens: int, page_size: int, seed: int) -> str:
        """Generate a unique prompt with deterministic prefix.

        The first `page_size` tokens are guaranteed to be unique for the
        provided seed. This enables KV-cache sharing across session turns.

        Args:
            num_tokens: Target total token count.
            page_size: Number of unique prefix tokens.
            seed: Seed for deterministic generation.

        Returns:
            Generated prompt text.
        """
        local_rng = random.Random(seed)
        vocab = self._sorted_vocab  # Use pre-cached sorted vocab

        digits = base10_to_basen(seed, len(vocab))
        tokens = [vocab[i] for i in digits]

        # Fill remaining with random tokens - use local_rng for determinism
        # We start from len(tokens) so we only add what's needed for page_size
        while len(tokens) < page_size:
            i = local_rng.randint(0, len(vocab) - 1)
            tokens.append(vocab[i])

        # Fill remaining up to num_tokens with random tokens
        target_len = num_tokens + 50
        while len(tokens) < target_len:
            i = local_rng.randint(0, len(vocab) - 1)
            tokens.append(vocab[i])

        return gen_prompt_from_corpus(
            num_tokens=num_tokens,
            pretokenized_lines=[tokens],
            tokenizer_handle=self.tokenizer,
            rng=local_rng,  # for shuffle, no-op for 1 line
        )

    def sample_from_corpus(self, num_tokens: int) -> str:
        """Sample tokens from the pre-tokenized corpus.

        Args:
            num_tokens: Target number of tokens to sample.

        Returns:
            Sampled text from corpus.
        """
        if num_tokens <= 0:
            return ""
        if not self.pretokenized_lines:
            raise ValueError("No corpus loaded for sampling")

        token_lines = [t for t in self.pretokenized_lines if t]
        if not token_lines:
            raise ValueError("All pretokenized_lines are empty.")

        remaining = num_tokens
        out: List[int] = []
        indices = list(range(len(token_lines)))
        self.rng.shuffle(indices)
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
                self.rng.shuffle(indices)

        return self.tokenizer.decode(out, skip_special_tokens=False)

    def build_from_hash_ids(
        self,
        hash_ids: List[int],
        block_size: int = 511,
        special_block_sizes: Optional[dict[int, int]] = None,
    ) -> str:
        """Build prompt from hash IDs (for MooncakeConv traces).

        Each hash_id maps to a deterministic block of tokens.

        Args:
            hash_ids: List of hash IDs.
            block_size: Default token block size per hash ID.
            special_block_sizes: Optional mapping of hash_id to specific block size.

        Returns:
            Generated prompt text.
        """
        prompt_parts = []
        special_block_sizes = special_block_sizes or {}

        for hash_id in hash_ids:
            current_size = special_block_sizes.get(hash_id, block_size)

            part = self.generate_unique_prompt(
                num_tokens=current_size,
                page_size=current_size,
                seed=hash_id,
            )
            prompt_parts.append(part)
        return "".join(prompt_parts)
