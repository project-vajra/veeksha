"""Corpus-free prompt generator backed by cached stable token encodings.

Design
------
A "stable encoding" is a token sequence E such that:

    encode(decode(E + E)) == E + E

i.e. the tokenizer round-trips the sequence stably when it appears next to
itself.  Because every encoding in our pool starts and ends with the same
boundary token (a leading/trailing space), the boundary between *any* two
encodings A and B looks identical to the A+A boundary — so A+B is also
stable.  This means we can tile encodings freely and decode the concatenated
token IDs exactly once to get a string that the server will re-tokenize to
approximately the same token count.

Performance
-----------
  First run   : generate N stable encodings (~2 s), save to disk.
  Later runs  : load from disk (~10 ms), pre-decode each (~0.1 ms × N).
  Per session : rng.choices(decoded_strings, k) + "".join  (~0.5 ms,
                no tokenizer call).

Token count accuracy: ±max_enc_length tokens (typically ±4).  For a 28 k-
token benchmark prompt that is ±0.01 % — irrelevant for throughput measurement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import List, Optional

from veeksha.core.tokenizer import TokenizerHandle

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path.home() / ".cache" / "veeksha" / "stable_encodings"
_NUM_ENCODINGS = 100
_GEN_SEED = 42  # fixed so the cache is reproducible across machines


def _tokenizer_fingerprint(handle: TokenizerHandle) -> str:
    """Short hash derived from encoding a handful of test strings.

    Used as a cache key so the cached encodings are invalidated automatically
    if the tokenizer changes.  The 5 encode calls take ~50 ms and only run
    once (the result is baked into the filename).
    """
    probes = [f" {v} " for v in (1, 42, 1337, 31415, 271828)]
    raw = str([list(handle.encode(p)) for p in probes])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _generate_stable_encodings(handle: TokenizerHandle, n: int) -> List[List[int]]:
    """Generate n stable token-ID sequences.

    Fast path: scan the vocabulary for single-token encodings that satisfy
    the stability property (encode(decode([t, t])) == [t, t]).  The vocabulary
    scan is O(vocab_size) with one decode + one encode per candidate and
    short-circuits as soon as n stable tokens are found (typically a few
    hundred ms).

    Fallback: if get_vocab is unavailable, probe random integer strings.
    This is much slower (~minutes) due to low pass rates from BPE boundary
    merges, but produces multi-token encodings with higher avg_len.
    """
    rng = random.Random(_GEN_SEED)

    # ── Fast path: vocabulary scan ─────────────────────────────────────
    if handle.get_vocab is not None:
        vocab: List[int] = list(handle.get_vocab())
        rng.shuffle(vocab)
        result: List[List[int]] = []
        for tok_id in vocab:
            enc = [tok_id]
            text = handle.decode(enc)
            if not text:
                continue
            doubled = enc + enc
            if list(handle.encode(handle.decode(doubled))) == doubled:
                result.append(enc)
                if len(result) >= n:
                    return result
        if len(result) >= n:
            return result
        logger.warning(
            "Only found %d/%d stable tokens via vocab scan; "
            "falling back to integer probe for remainder",
            len(result),
            n,
        )
    else:
        result = []

    # ── Slow fallback: probe random integer strings ────────────────────
    used: set[int] = set()
    max_attempts = n * 100_000
    for _ in range(max_attempts):
        if len(result) >= n:
            break
        seed = rng.randint(0, 10_000_000)
        if seed in used:
            continue
        used.add(seed)
        tokens = list(handle.encode(f" {seed} "))
        if not tokens:
            continue
        doubled = tokens + tokens
        if list(handle.encode(handle.decode(doubled))) == doubled:
            result.append(tokens)

    if len(result) < n:
        logger.warning(
            "Could only find %d/%d stable encodings after %d attempts",
            len(result),
            n,
            max_attempts,
        )
    return result


def _load_or_generate(handle: TokenizerHandle) -> List[List[int]]:
    """Return cached stable encodings, generating and caching them if needed."""
    fp = _tokenizer_fingerprint(handle)
    cache_path = _CACHE_ROOT / f"{fp}.json"

    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            if (
                isinstance(data, list)
                and len(data) >= _NUM_ENCODINGS
                and all(isinstance(enc, list) for enc in data[:_NUM_ENCODINGS])
            ):
                logger.debug(
                    "Loaded %d stable encodings from %s", len(data), cache_path
                )
                return data[:_NUM_ENCODINGS]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt cache at %s (%s), regenerating", cache_path, exc)

    logger.info(
        "Generating %d stable prompt encodings for this tokenizer "
        "(one-time cost, ~2s — result cached to %s)",
        _NUM_ENCODINGS,
        cache_path,
    )
    encodings = _generate_stable_encodings(handle, _NUM_ENCODINGS)
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(encodings))
    logger.info("Stable encodings cached to %s", cache_path)
    return encodings


class PromptStringGenerator:
    """Generate unique prompts via cached stable encodings.

    All tokenizer work is done at construction time.  generate() performs
    only Python list/string operations (no tokenizer calls).

    Args:
        tokenizer_handle: The model's tokenizer handle.
        rng: Seeded random.Random instance for reproducibility.
    """

    def __init__(
        self,
        tokenizer_handle: TokenizerHandle,
        rng: Optional[random.Random] = None,
    ):
        self._rng = rng or random.Random()

        token_encodings = _load_or_generate(tokenizer_handle)

        # Pre-decode every encoding once (O(1) per encoding, ~0.1 ms each)
        self._decoded: List[str] = [
            tokenizer_handle.decode(enc) for enc in token_encodings
        ]
        self._enc_lengths: List[int] = [len(enc) for enc in token_encodings]
        self._avg_len: float = sum(self._enc_lengths) / len(self._enc_lengths)

        logger.debug(
            "PromptStringGenerator ready: %d encodings, avg %.1f tokens each",
            len(self._decoded),
            self._avg_len,
        )

    def generate(self, num_tokens: int) -> str:
        """Return a string of approximately num_tokens tokens.

        Tiles randomly chosen stable decoded strings and joins them.
        Token count accuracy: ±max_enc_length (typically ±4 tokens).
        No tokenizer calls — pure Python string operations.
        """
        if num_tokens <= 0:
            return ""
        k = max(1, round(num_tokens / self._avg_len))
        return "".join(self._rng.choices(self._decoded, k=k))
