# Native ASR/STT Probe Design Handoff

This note is a handoff for continuing the ASR/STT capacity-analysis work in this
repo. It captures the current benchmark shape, why the Python client path is
suspect at high concurrency, and a concrete design for a native
dispatch/receive/timestamping probe written in C++ with Boost.Beast.

## Current Code Path

Relevant files already read:

- `docs/ASR_BENCHMARKING.md`
- `guide.md`
- `scripts/prepare_audio_traces.py`
- `scripts/sweep.py`
- `scripts/plot_sweep.py`
- `scripts/utils.py`
- `veeksha/sweeps/planner.py`
- `configs/stt_vajra.yaml`
- `configs/stt_vllm_realtime.yaml`
- `veeksha/client/stt.py`
- `veeksha/config/client.py`
- `veeksha/generator/session/trace/audio.py`
- `veeksha/config/generator/session.py`
- `veeksha/evaluator/performance/asr.py`
- `veeksha/evaluator/performance/audio.py`
- `veeksha/evaluator/performance/base.py`
- `veeksha/evaluator/registry.py`
- `veeksha/client/registry.py`
- `veeksha/benchmark.py`
- `veeksha/benchmark_utils.py`
- `veeksha/workers/dispatch.py`
- `veeksha/workers/completion.py`
- `veeksha/workers/client_runner.py`
- `veeksha/workers/prefetch.py`
- `veeksha/traffic/concurrent.py`
- `veeksha/cli/benchmarks.py`
- `veeksha/sweep_summary.py`
- `veeksha/config/utils.py`
- `veeksha/config/core/flat_dataclass.py`

Current flow:

1. `scripts/prepare_audio_traces.py` creates `traces/asr/aa_public/manifest.jsonl`
   plus WAV files from public ArtificialAnalysis/VoxPopuli and Earnings22 inputs.
2. `AudioTraceFlavorGenerator` emits one AUDIO request per manifest row,
   preserving metadata such as `expected_transcript`, dataset, parent ID, and
   chunk ordering.
3. `STTClient` in `veeksha/client/stt.py` streams PCM16 mono over WebSocket for
   Vajra `/stream` or vLLM Realtime `/v1/realtime`.
4. `AudioPerformanceEvaluator` writes request-level metrics and delegates WER to
   `ASRMetricAccumulator`.
5. `ASRMetricAccumulator` computes clip WER directly and regrouped parent-level
   WER for Earnings22 chunks.
6. Sweep entry points are `scripts/sweep.py` / `veeksha.sweeps.planner`, plus the
   generic YAML `!expand` sweep path.

The repo aims to do capacity analysis by fixing the model/workload and sweeping
engine/concurrency, watching `Error Rate`, `ttfc`, `time_to_first_partial`,
`time_to_final_transcript`, `end_to_end_latency`, `rtf`, and
`asr_final_*_wer`.

## Problem

The current Python client path is not trustworthy at 1000s of concurrent
sessions. The observed timestamp is taken only after Python's event loop/thread
wakes up, receives the WebSocket message, and often after Python runtime
overhead. At high concurrency this can contaminate the measured latency
distribution.

Known Python-side contamination risks:

- `websockets` receive scheduling delay.
- JSON parse and object allocation under Python runtime overhead.
- Many Python threads/event loops in the current worker model.
- `asyncio.sleep` pacing jitter for realtime audio.
- Synchronous audio load/resample work in `_audio_to_pcm16_bytes`.
- Queue/dispatch/completion worker overhead.
- GIL effects and large per-session Python object graphs.

For controlled servers like Vajra/vLLM, server-side timestamps can help, but this
does not generalize to closed providers like OpenAI or ElevenLabs. For closed
APIs, we still need high-quality client-observed timestamps.

## Recommendation

Build a native C++ probe as a separate executable, not a `pybind` hot path.

Python should remain the orchestrator/analyzer:

- expand configs and sweeps
- choose manifest rows
- generate a native run config
- launch the native probe as a subprocess
- read native event/metric output
- compute WER
- plot and summarize

The native probe should own:

- WebSocket connect/auth
- audio pacing
- send timestamping
- receive timestamping
- minimal provider event parsing
- request/session error classification
- probe health telemetry

Use Boost.Beast on top of Boost.Asio for the transport. Beast gives us explicit
control over async WebSocket reads/writes, TLS, timers, strands, connection
lifecycle, and buffer ownership while keeping the implementation in modern C++.

## C++ / Boost.Beast Design

Use asynchronous Boost.Asio throughout. Do not use one thread per session.

Core model:

- Create `N` event-loop shards.
- Each shard owns one `boost::asio::io_context` and one worker thread.
- Assign sessions to shards by round-robin or least-loaded shard.
- Each WebSocket session is owned by exactly one shard.
- Each session owns its resolver, TCP/TLS/WebSocket stream, timers, read buffer,
  write queue, provider state, and request metric state.
- Run one continuous async read chain per session.
- Serialize all writes through one per-session outbound queue.

If a shard ever runs an `io_context` on multiple threads, wrap each session in a
strand. The default design should instead prefer one thread per shard and many
sessions per thread. This keeps session state single-threaded and makes hot-path
timestamping easier to reason about.

Transport types:

```cpp
using tcp = boost::asio::ip::tcp;
using TlsStream = boost::beast::ssl_stream<boost::beast::tcp_stream>;
using WsStream = boost::beast::websocket::stream<TlsStream>;
```

For local `ws://` validation, also support a plain stream variant:

```cpp
using PlainWsStream =
    boost::beast::websocket::stream<boost::beast::tcp_stream>;
```

Provider protocol code must be isolated from the scheduler/timing core. The
runner should not know whether an audio chunk becomes raw binary, JSON with
Base64 audio, or a provider-specific envelope.

Provider interface sketch:

```cpp
struct OutboundMessage {
  std::string payload;
  bool binary = false;
  std::optional<int64_t> audio_offset_ms;
};

struct ProviderEvent {
  std::string event_type;
  std::string provider_event_id;
  std::string provider_item_id;
  std::optional<std::string> text_delta;
  std::optional<std::string> final_text;
  std::optional<std::string> error_code;
  std::optional<std::string> error_message;
  bool is_partial = false;
  bool is_final = false;
  bool is_error = false;
};

class Provider {
 public:
  virtual ~Provider() = default;
  virtual std::string_view Name() const = 0;
  virtual std::vector<OutboundMessage> BuildStartSession() = 0;
  virtual OutboundMessage BuildAudioChunk(
      std::string_view chunk,
      int64_t audio_offset_ms) = 0;
  virtual std::optional<OutboundMessage> BuildCommit() = 0;
  virtual ProviderEvent ParseEvent(std::string_view frame) = 0;
};
```

Session loop sketch:

```text
resolve
tcp connect
tls handshake, if wss
websocket handshake
send provider session-init messages
start async_read loop
schedule audio chunks with steady_timer
for each chunk:
  record send_scheduled_ns
  when timer fires, record send_actual_ns
  enqueue websocket write
commit/end input
wait for final transcript or timeout
write request metric
close
```

Important Beast constraints:

- Do not issue overlapping `async_write` calls on one stream.
- Do not mutate a buffer until the async operation using it has completed.
- The read handler timestamp is the earliest application-level receive timestamp
  Beast exposes. It is not a kernel packet timestamp.
- Timestamp at the first instruction in the read handler, before JSON parse,
  transcript assembly, logging, or metric aggregation.
- Disable WebSocket compression unless a provider explicitly requires it.
- Set timeouts explicitly on resolve/connect/handshake/request/finalization.

## Dependency And Build Plan

The repo is Python-first, so the native probe should be an isolated CMake
subproject under `native/stt_probe_cpp/`.

Use `mamba` for local C++ dependencies whenever possible:

```bash
mamba install -c conda-forge cmake ninja boost-cpp nlohmann_json openssl
```

This repo mirrors Vajra's path-based environment convention. Prefer the checked
in environment file:

```bash
make setup/environment
conda activate ./env
```

The probe should not rely on system headers being present. CMake should discover
dependencies from the active conda/mamba prefix via `find_package`.

Initial build command:

```bash
cmake -S native/stt_probe_cpp -B native/stt_probe_cpp/build -G Ninja
cmake --build native/stt_probe_cpp/build
```

Native C++ style targets mirror the Vajra workflow:

```bash
make format/cpp
make lint/cpp
```

## Provider Notes

OpenAI Realtime:

- WebSocket is the low-level server-side transport.
- Client sends JSON events and Base64-encoded audio chunks.
- Realtime transcription uses `input_audio_buffer.append`; if turn detection is
  disabled, the client commits with `input_audio_buffer.commit`.
- Current OpenAI Realtime transcription examples use mono PCM input such as
  `audio/pcm`.
- Transcription emits delta and completed events; ordering between completion
  events from different turns is not guaranteed, so use item/session IDs.

ElevenLabs:

- Realtime STT endpoint: `wss://api.elevenlabs.io/v1/speech-to-text/realtime`.
- Sends `input_audio_chunk`; receives partial/committed transcript events.
- TTS WebSocket endpoint: `/v1/text-to-speech/{voice_id}/stream-input`.
- TTS WebSocket returns audio and alignment payloads.
- Their concurrency model is provider-specific: HTTP requests count
  individually, but for WebSocket TTS only generation time counts against
  concurrency.
- 429 can mean rate limit or concurrency limit; error codes distinguish
  `rate_limit_exceeded` vs `concurrent_limit_exceeded`.

## Native Probe CLI Shape

Proposed executable name:

```bash
veeksha-native-stt \
  --run-config /path/to/native_run.json \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/run_dir \
  --provider vajra|vllm_realtime|openai_realtime|elevenlabs_realtime_stt \
  --concurrency 1000 \
  --duration-seconds 300
```

The native run config should be generated by Python from the existing benchmark
config, not hand-authored.

Suggested `native_run.json` fields:

```json
{
  "provider": "openai_realtime",
  "endpoint": "wss://api.openai.com/v1/realtime?model=...",
  "headers": {
    "Authorization": "Bearer ${OPENAI_API_KEY}"
  },
  "audio": {
    "sample_rate_hz": 24000,
    "channels": 1,
    "sample_format": "pcm16",
    "chunk_ms": 100,
    "realtime_pacing": true
  },
  "session": {
    "connect_timeout_ms": 10000,
    "request_timeout_ms": 120000,
    "turn_detection": "manual"
  },
  "load": {
    "concurrency": 1000,
    "ramp_seconds": 120,
    "duration_seconds": 300,
    "seed": 1234
  },
  "trace": {
    "write_events": true,
    "write_raw_provider_events": false,
    "profile": true
  }
}
```

## Output Files

Write all output into the normal run directory so existing Python evaluators can
consume it.

Required:

- `native_events.jsonl`: lossless event stream from native probe.
- `native_request_metrics.jsonl`: one row per logical ASR request.
- `native_probe_health.json`: run-level probe health and saturation indicators.
- `native_errors.jsonl`: structured connect/send/recv/protocol/provider errors.

Optional:

- `native_summary.json`
- `perf.data` or `perf script` output
- allocator or heap profiling output, if enabled in the local build

## Event Schema

The key design rule: timestamp first, parse second.

For receive events, take `rx_frame_at_ns` as soon as the Beast read handler is
entered, before JSON parse, transcript accumulation, logging, or metric
aggregation.

Suggested `native_events.jsonl` record:

```json
{
  "schema_version": 1,
  "run_id": "2026-06-01T...",
  "session_id": "s-000001",
  "request_id": "r-000001",
  "provider": "openai_realtime",
  "event_seq": 42,
  "event_type": "rx_frame|parsed_event|send_audio|send_commit|connect|close|error",
  "monotonic_ns": 123456789,
  "wall_unix_ns": 1780000000000000000,
  "send_scheduled_ns": 123400000,
  "send_actual_ns": 123401234,
  "rx_frame_at_ns": 123456789,
  "parsed_at_ns": 123457100,
  "provider_event_type": "conversation.item.input_audio_transcription.delta",
  "provider_event_id": "evt_...",
  "provider_item_id": "item_...",
  "audio_offset_ms": 1200,
  "text_delta": "hello",
  "final_text": null,
  "error_code": null,
  "error_message": null
}
```

Suggested `native_request_metrics.jsonl` record:

```json
{
  "schema_version": 1,
  "request_id": "r-000001",
  "session_id": "s-000001",
  "dataset": "aa_voxpopuli",
  "parent_id": null,
  "chunk_index": null,
  "audio_duration_ms": 8340,
  "connect_start_ns": 1000,
  "connect_done_ns": 2000,
  "audio_send_start_ns": 3000,
  "audio_send_done_ns": 4000,
  "commit_sent_ns": 4100,
  "first_rx_frame_ns": 5000,
  "first_partial_ns": 5200,
  "final_transcript_ns": 10000,
  "close_ns": 11000,
  "send_drift_p50_ms": 0.4,
  "send_drift_p99_ms": 3.2,
  "rx_parse_delay_p50_ms": 0.1,
  "rx_parse_delay_p99_ms": 0.9,
  "final_transcript": "hello world",
  "status": "ok|timeout|connect_error|provider_error|protocol_error"
}
```

## Probe Health Schema

`native_probe_health.json` should make it obvious when the client is the
bottleneck.

Include:

- max active sessions
- total connects / failed connects
- total frames received
- total frames sent
- CPU usage over time
- RSS over time
- thread count over time
- event writer backlog summary
- per-shard event-loop lag
- send pacing drift p50/p90/p99/max
- receive parse delay p50/p90/p99/max
- dropped event count
- JSON parse failure count
- provider 429 / throttling / close-code counts

If these health metrics show client pressure, treat the benchmark as invalid.

## Timing Semantics

Use two clocks:

- `std::chrono::steady_clock` for latency deltas within the native process
- `std::chrono::system_clock` Unix nanoseconds for correlating external logs

Do not compute elapsed times from wall-clock timestamps.

Metric suggestions:

- `ttfc`: request/session start to first received provider content frame,
  preserving existing repo semantics where possible.
- `time_to_first_partial`: request/session start or post-commit start to first
  transcript delta. Record both if possible.
- `time_to_final_transcript`: request/session start or post-commit start to final
  transcript. Record both if possible.
- `send_pacing_drift`: `send_actual_ns - send_scheduled_ns` per chunk.
- `rx_parse_delay`: `parsed_at_ns - rx_frame_at_ns`.
- `event_loop_lag`: measured by per-shard heartbeat timers that compare expected
  timer fire time to actual handler entry time.

For closed providers, these are still client-observed timestamps. They do not
expose server queue time unless the provider exposes it in events/headers.

## Hot Path Rules

- Pre-convert/resample all audio before the run.
- Preload audio bytes or memory-map them; do not decode audio in the send loop.
- Avoid logging per frame synchronously.
- Use a dedicated event writer with bounded queues and dropped-event counters.
- Timestamp before JSON parse.
- Parse only the fields needed for timing and transcript assembly.
- Store raw provider events only in a debug mode; raw event logging can dominate.
- Disable WebSocket compression unless specifically testing it.
- Jitter/ramp session starts to avoid artificial handshake storms.
- Keep provider message builders allocation-aware; reuse buffers where practical.

## OS/System Breakpoints

Expect these to matter at 1000+ sessions:

- `ulimit -n` / file descriptor limits
- ephemeral port range and `TIME_WAIT`
- TLS handshake CPU
- DNS and proxy behavior
- socket send/receive buffer sizing
- local NIC bandwidth and queueing
- NAT/firewall/conntrack limits
- provider WebSocket connection caps
- provider rate/concurrency limits
- JSON/Base64 CPU cost
- event-log disk throughput

Linux socket timestamping may help transport debugging, but it is not a
replacement for application-level WebSocket event timestamps. With TLS and
byte-stream WebSockets, correlating kernel packet/byte timestamps to logical
transcript events is nontrivial.

## Integration Plan

Milestone 1: Native C++ sidecar, no repo-wide refactor

1. Add `native/stt_probe_cpp/`.
2. Add a CMake build that discovers Boost, OpenSSL, Threads, and JSON headers from
   the active mamba/conda environment.
3. Build config parsing, manifest parsing, event writing, health snapshot
   writing, provider selection, and PCM16 WAV extraction.
4. Implement Boost.Beast live streaming for one local provider first, probably
   `vllm_realtime` or `vajra`, because local validation is easier.
5. Add a small Python adapter that can call the binary and copy/convert its
   output into the existing run directory.
6. Reuse the existing ASR evaluator for WER from final transcripts.

Milestone 2: Provider adapters

1. Implement provider-specific WebSocket event builders/parsers:
   - Vajra
   - vLLM Realtime
   - OpenAI Realtime transcription
   - ElevenLabs realtime STT
2. Keep provider protocol code isolated from the scheduler/timing core.

Milestone 3: Sweep integration

1. Add config flag such as `client.execution_backend: python|native_cpp`.
2. In sweep runs, route native configs through the subprocess adapter.
3. Preserve current output naming so `sweep_summary.json` and plots keep working.
4. Add native probe health into summary artifacts.

Milestone 4: Validation

1. Run Python and native probes at low concurrency against local servers; metrics
   should roughly match.
2. Run both at increasing concurrency; native should show lower probe-side lag.
3. Add a fake/local WebSocket STT server that emits deterministic delayed events
   to measure client timestamp error directly.
4. Validate that send pacing drift, rx parse delay, and per-shard event-loop lag
   remain small at target load.

## Proposed Internal C++ Structure

```text
native/stt_probe_cpp/
  CMakeLists.txt
  README.md
  src/Main.cpp
  src/config/Config.h
  src/config/Config.cpp
  src/manifest/Manifest.h
  src/manifest/Manifest.cpp
  src/provider/Provider.h
  src/provider/ProviderRegistry.h
  src/provider/Vajra.h
  src/provider/Vajra.cpp
  src/provider/VllmRealtime.h
  src/provider/VllmRealtime.cpp
  src/runner/Scheduler.h
  src/runner/Scheduler.cpp
  src/trace/Writer.h
  src/trace/Writer.cpp
  src/health/Health.h
  src/health/Health.cpp
  src/util/Cli.h
  src/util/Cli.cpp
```

## Acceptance Criteria

The native probe is useful only if each benchmark run can answer:

- Did the client saturate?
- How much send pacing drift occurred?
- How long did received frames wait before parsing?
- Were provider errors/rate limits separate from client errors?
- Are low-concurrency metrics comparable to the existing Python client?
- Does high-concurrency client health remain clean when provider latency worsens?

Do not trust high-concurrency latency numbers unless `native_probe_health.json`
shows low per-shard event-loop lag, low send drift, low rx parse delay, no event
writer drops, and acceptable CPU/RSS behavior.
