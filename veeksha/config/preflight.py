"""Configuration for the ``veeksha preflight`` command.

Preflight certifies that the benchmark harness keeps time honestly before you
trust its numbers: it drives the real scheduler -> dispatch -> client path
against a deterministic mock server and gates the p99 timing drift (request /
response delivery, server pacing fidelity, dispatch drift) against thresholds.

Workload/timing is grouped by client category -- ``text`` (chat, completions),
``tts`` (tts, streaming_tts), and ``stt`` -- because the response shape differs
(token stream vs audio chunks vs transcript deltas). The streaming-TTS check
runs once per wire protocol the client supports, since each protocol has its own
framing (base64-in-JSON vs binary) and response-trigger rules.

Traffic is closed-loop concurrency: ``concurrency`` sessions in flight out of
``runtime.max_sessions`` total, built with ``rampup_seconds=0`` so measurement
runs at steady state.
"""

from typing import List, Tuple

from vidhi import field, frozen_dataclass

from veeksha.cli.base import VeekshaCommand
from veeksha.config.runtime import RuntimeConfig
from veeksha.config.traffic import BaseTrafficConfig, ConcurrentTrafficConfig

# Every check preflight can run, in execution order. These double as the report
# subdirectory names, so they are the identifiers users select by.
PREFLIGHT_CHECKS: Tuple[str, ...] = (
    "chat",
    "completions",
    "tts",
    "streaming_tts_openai",
    "streaming_tts_vajra",
    "stt",
)


@frozen_dataclass
class PreflightTextCheckConfig:
    """Workload + mock timing for the text checks (chat, completions).

    Input is a single-shot text prompt; the mock streams a token response.
    """

    input_tokens: int = field(100, help="Prompt length in tokens.")
    num_response_chunks: int = field(
        100, help="Response tokens the mock emits (completions emits one response)."
    )
    server_ttfc_ms: float = field(200.0, help="Mock time-to-first-chunk delay (ms).")
    server_tpoc_ms: float = field(
        20.0, help="Mock time-per-output-chunk (inter-chunk) delay (ms)."
    )


@frozen_dataclass
class PreflightTtsCheckConfig:
    """Workload + mock timing for the TTS checks (tts, streaming_tts).

    Input is a text prompt (streamed in paced segments for the WebSocket
    client); the mock streams an audio-chunk response.
    """

    input_tokens: int = field(100, help="Prompt length in tokens.")
    input_chunk_tokens: int = field(
        4, help="Tokens per streamed input message (streaming_tts)."
    )
    input_pacing_tps: float = field(
        50.0, help="Input pacing rate in tokens/sec (streaming_tts)."
    )
    num_response_chunks: int = field(100, help="Audio chunks the mock emits.")
    server_ttfc_ms: float = field(200.0, help="Mock time-to-first-chunk delay (ms).")
    server_tpoc_ms: float = field(20.0, help="Mock inter-audio-chunk delay (ms).")


@frozen_dataclass
class PreflightSttCheckConfig:
    """Workload + mock timing for the STT check.

    Input is streamed audio; the mock streams a transcript-delta response.
    """

    input_seconds: float = field(3.0, help="Generated audio clip length in seconds.")
    input_chunk_bytes: int = field(4096, help="Audio bytes per streamed input message.")
    sample_rate: int = field(16000, help="Audio sample rate in Hz.")
    num_response_chunks: int = field(40, help="Transcript deltas the mock emits.")
    server_ttfc_ms: float = field(200.0, help="Mock time-to-first-chunk delay (ms).")
    server_tpoc_ms: float = field(50.0, help="Mock inter-transcript-delta delay (ms).")


@frozen_dataclass
class PreflightCheckConfig(VeekshaCommand, name="preflight"):
    """Certify measurement fidelity of the harness at a target concurrency."""

    # --- which client pathways to exercise ---
    checks: List[str] = field(
        default_factory=list,
        help=(
            "Checks to run, space separated; omit to run all of them. Choices: "
            + " ".join(PREFLIGHT_CHECKS)
        ),
    )

    # --- traffic (closed-loop concurrency; scheduler built with rampup=0) ---
    concurrency: int = field(
        50,
        help="Target number of concurrent sessions to sustain (the load level).",
    )
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(max_sessions=500),
        help=(
            "Harness runtime knobs for the run. ``max_sessions`` is the total "
            "sessions to drive (the sample size); the thread-pool sizes "
            "(``num_client_threads``, ``num_dispatcher_threads``, "
            "``num_completion_threads``) are also available for tuning."
        ),
    )

    # --- per-category workload + mock timing ---
    text: PreflightTextCheckConfig = field(default_factory=PreflightTextCheckConfig)
    tts: PreflightTtsCheckConfig = field(default_factory=PreflightTtsCheckConfig)
    stt: PreflightSttCheckConfig = field(default_factory=PreflightSttCheckConfig)

    # --- gate thresholds (p99, in milliseconds unless noted) ---
    delivery_lag_threshold_ms: float = field(
        10.0,
        aliases=["delivery-lag-threshold-ms"],
        help=(
            "Max allowed p99 for request/response/input delivery lag (client<->"
            "server). Above this the harness is adding transport/queueing drift."
        ),
    )
    server_pacing_threshold_ms: float = field(
        10.0,
        aliases=["server-pacing-threshold-ms"],
        help=(
            "Max allowed p99 for the mock server's own ttfc/tpoc pacing error. "
            "If exceeded the server itself is the bottleneck (SERVER_AT_CAPACITY)."
        ),
    )
    dispatch_drift_threshold_ms: float = field(
        10.0,
        aliases=["dispatch-drift-threshold-ms"],
        help=(
            "Max allowed p99 for end-to-end dispatch drift (client-sent minus "
            "scheduled-ready). Above this the harness dispatches off-schedule."
        ),
    )
    input_pacing_threshold_ms: float = field(
        10.0,
        aliases=["input-pacing-threshold-ms"],
        help=(
            "Streaming-input clients only. Max allowed p99 for input pacing "
            "error (actual segment send time minus its intended deadline)."
        ),
    )
    max_unpaired_fraction: float = field(
        0.0,
        aliases=["max-unpaired-fraction"],
        help=(
            "Max fraction of requests allowed to lack a matching server record. "
            "Defaults to 0: an unpaired request means a timeline went unmeasured."
        ),
    )

    # --- output ---
    output_dir: str = field(
        "preflight_report",
        aliases=["output-dir"],
        help="Directory for the rendered preflight report.",
    )

    def build_traffic(self) -> BaseTrafficConfig:
        """Closed-loop scheduler at the target concurrency, no ramp.

        rampup_seconds is pinned to 0: preflight measures steady-state drift, so
        it must be at full concurrency from the start.
        """
        return ConcurrentTrafficConfig(
            target_concurrent_sessions=self.concurrency, rampup_seconds=0
        )

    def selected_checks(self) -> Tuple[str, ...]:
        """Checks to run; an empty selection means every check.

        Always in ``PREFLIGHT_CHECKS`` order, so run order and report layout do
        not depend on the order they were listed in, and repeats collapse.
        """
        if not self.checks:
            return PREFLIGHT_CHECKS
        selected = set(self.checks)
        return tuple(name for name in PREFLIGHT_CHECKS if name in selected)

    def __post_init__(self) -> None:
        unknown = [name for name in self.checks if name not in PREFLIGHT_CHECKS]
        if unknown:
            raise ValueError(
                f"Unknown preflight check(s): {' '.join(unknown)}. "
                f"Choices: {' '.join(PREFLIGHT_CHECKS)}"
            )
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.runtime.max_sessions <= 0:
            raise ValueError("runtime.max_sessions must be positive")
        if not 0.0 <= self.max_unpaired_fraction <= 1.0:
            raise ValueError("max_unpaired_fraction must be in [0, 1]")
        for name in (
            "delivery_lag_threshold_ms",
            "server_pacing_threshold_ms",
            "dispatch_drift_threshold_ms",
            "input_pacing_threshold_ms",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
