"""QUALITY-mode enforcement: the mode must change behaviour, loudly.

Every check here exists because its failure shape is silent: a wrapped trace,
a max_sessions above or below the corpus, or a QUALITY flag that validates
nothing all complete cleanly and report a number that is not the corpus WER.
The PERFORMANCE cases pin the disabled path: setting the mode off must leave
those same configurations accepted.
"""

import tempfile
import unittest
from pathlib import Path

from veeksha.config.benchmark import BenchmarkConfig
from veeksha.config.generator.session import TraceSessionGeneratorConfig
from veeksha.config.runtime import RuntimeConfig
from veeksha.types import BenchmarkMode


class TestQualityModeEnforcement(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.trace = Path(self._tmp.name) / "manifest.jsonl"
        self.trace.write_text(
            '{"session_id": 1}\n{"session_id": 2}\n{"session_id": 3}\n'
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _config(self, mode, wrap_mode=False, max_sessions=3, trace_file=None):
        return BenchmarkConfig(
            mode=mode,
            session_generator=TraceSessionGeneratorConfig(
                trace_file=str(trace_file or self.trace), wrap_mode=wrap_mode
            ),
            runtime=RuntimeConfig(max_sessions=max_sessions),
        )

    # ----- enabled path: QUALITY must reject every silent-corruption shape -----

    def test_quality_accepts_one_pass_over_the_corpus(self):
        config = self._config(BenchmarkMode.QUALITY)
        self.assertIs(config.mode, BenchmarkMode.QUALITY)

    def test_quality_rejects_wrap_mode(self):
        with self.assertRaisesRegex(ValueError, "wrap_mode=False"):
            self._config(BenchmarkMode.QUALITY, wrap_mode=True)

    def test_quality_rejects_max_sessions_below_corpus(self):
        with self.assertRaisesRegex(ValueError, "3 sessions"):
            self._config(BenchmarkMode.QUALITY, max_sessions=2)

    def test_quality_rejects_max_sessions_above_corpus(self):
        with self.assertRaisesRegex(ValueError, "3 sessions"):
            self._config(BenchmarkMode.QUALITY, max_sessions=4)

    def test_quality_rejects_unlimited_sessions(self):
        with self.assertRaisesRegex(ValueError, "max_sessions == corpus size"):
            self._config(BenchmarkMode.QUALITY, max_sessions=-1)

    def test_quality_rejects_missing_trace_file(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self._config(
                BenchmarkMode.QUALITY, trace_file=self.trace.with_name("no.jsonl")
            )

    def test_quality_rejects_empty_trace_file(self):
        empty = self.trace.with_name("empty.jsonl")
        empty.write_text("\n")
        with self.assertRaisesRegex(ValueError, "no sessions"):
            self._config(BenchmarkMode.QUALITY, trace_file=empty)

    def test_quality_rejects_non_trace_generator(self):
        with self.assertRaisesRegex(ValueError, "trace-driven"):
            BenchmarkConfig(
                mode=BenchmarkMode.QUALITY,
                runtime=RuntimeConfig(max_sessions=3),
            )

    def test_quality_counts_csv_sessions_by_session_id(self):
        csv = self.trace.with_name("manifest.csv")
        csv.write_text("session_id\n1\n2\n")
        config = self._config(BenchmarkMode.QUALITY, max_sessions=2, trace_file=csv)
        self.assertIs(config.mode, BenchmarkMode.QUALITY)

    def test_quality_counts_sessions_not_rows_for_multi_turn_traces(self):
        # The generator builds sessions via groupby("session_id"): a
        # conversation trace with several rows per session is ONE session per
        # id. Counting lines would demand a max_sessions the generator can
        # never produce.
        multi = self.trace.with_name("multiturn.jsonl")
        multi.write_text(
            '{"session_id": 1, "turn": 1}\n{"session_id": 1, "turn": 2}\n'
            '{"session_id": 2, "turn": 1}\n'
        )
        config = self._config(BenchmarkMode.QUALITY, max_sessions=2, trace_file=multi)
        self.assertIs(config.mode, BenchmarkMode.QUALITY)
        with self.assertRaisesRegex(ValueError, "2 sessions"):
            self._config(BenchmarkMode.QUALITY, max_sessions=3, trace_file=multi)

    def test_quality_rejects_rows_without_session_id(self):
        bad = self.trace.with_name("bad.jsonl")
        bad.write_text('{"audio_file": "x.wav"}\n')
        with self.assertRaisesRegex(ValueError, "session_id"):
            self._config(BenchmarkMode.QUALITY, max_sessions=1, trace_file=bad)

    # ----- disabled path: PERFORMANCE must accept those same shapes -----

    def test_performance_accepts_wrap_mode(self):
        config = self._config(BenchmarkMode.PERFORMANCE, wrap_mode=True)
        self.assertIs(config.mode, BenchmarkMode.PERFORMANCE)

    def test_performance_accepts_mismatched_max_sessions(self):
        config = self._config(BenchmarkMode.PERFORMANCE, max_sessions=999)
        self.assertIs(config.mode, BenchmarkMode.PERFORMANCE)

    def test_performance_is_the_default(self):
        config = BenchmarkConfig()
        self.assertIs(config.mode, BenchmarkMode.PERFORMANCE)


if __name__ == "__main__":
    unittest.main()
