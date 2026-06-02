# Veeksha Native STT Probe

This is the C++/Boost.Beast sidecar for high-concurrency ASR/STT client-side
timestamping.

Create the repo-local environment from the repository root:

```bash
make setup/environment
conda activate ./env
```

Build:

```bash
cmake -S native/stt_probe_cpp -B native/stt_probe_cpp/build -G Ninja
cmake --build native/stt_probe_cpp/build
```

Format and lint the native C++ files from the repository root:

```bash
make format/cpp
make lint/cpp
```

Current state:

- Config parsing is implemented.
- ASR manifest parsing is implemented.
- PCM16 mono WAV extraction is implemented for prepared ASR traces.
- Vajra and vLLM Realtime provider message/parsing adapters are implemented.
- Native event, request-metric, error, and health writers are implemented.
- `--dry-run` validates the end-to-end file flow.
- Live Boost.Beast streaming supports `ws://` and `wss://`.
- Live mode runs up to `--concurrency` coroutine workers and writes
  `connect_error` / `protocol_error` / `send_error` request rows instead of
  crashing on per-request failures.

Validated:

- Built successfully with the repo-local `./env`.
- `make lint/cpp` passes.
- `--dry-run` works against `traces/asr/aa_public/manifest.jsonl`.
- Live vLLM Realtime e2e was validated on GPU 0 with
  `mistralai/Voxtral-Mini-4B-Realtime-2602` through Docker port `8025`.
- The live vLLM check completed `concurrency=2`, `max_requests=4` with zero
  native error rows and all request statuses `ok`.

What remains:

- Add a bounded asynchronous event writer. The current writer is synchronous,
  which is fine for correctness smoke tests but not acceptable for trusting
  high-concurrency latency numbers.
- Add ramp and duration semantics. Today `--concurrency` bounds worker count and
  `--max-requests` bounds total work; `duration_seconds` is parsed but not used
  as a run-stop condition.
- Integrate with Python orchestration. The next repo-level step is a Python
  adapter that generates `native_run.json`, invokes this binary, and copies or
  converts outputs into the normal benchmark run directory.
- Reuse the existing ASR/WER evaluator on `native_request_metrics.jsonl`.
- Validate against Vajra `/stream`, not just vLLM Realtime.
- Add OpenAI Realtime and ElevenLabs realtime STT provider adapters if closed
  provider testing is still in scope.
- Expand probe health telemetry: CPU/RSS sampling, per-worker event-loop lag,
  event writer backlog/drops, send drift p90/p99/max, and rx parse delay
  p90/p99/max.
- Tighten cancellation and timeout behavior under server stalls, half-closed
  sockets, and provider protocol errors.
- Add a deterministic fake WebSocket STT server for timestamp-error regression
  tests.

Example dry run:

```bash
native/stt_probe_cpp/build/veeksha-native-stt \
  --run-config /path/to/native_run.json \
  --manifest traces/asr/aa_public/manifest.jsonl \
  --output-dir /tmp/veeksha-native-dry-run \
  --provider vllm_realtime \
  --concurrency 4 \
  --duration-seconds 30 \
  --dry-run \
  --max-requests 2
```

Example live run:

```bash
native/stt_probe_cpp/build/veeksha-native-stt \
  --run-config native/stt_probe_cpp/examples/native_run_vllm_realtime.json \
  --manifest traces/asr/aa_public/manifest.jsonl \
  --output-dir /tmp/veeksha-native-live \
  --provider vllm_realtime \
  --concurrency 4 \
  --max-requests 8
```

vLLM server launch used for e2e validation:

```bash
docker run -d --rm --name veeksha-voxtral-e2e-gpu0 \
  --gpus '"device=0"' -p 8025:8025 --ipc=host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/aagrawal360/.cache/huggingface:/root/.cache/huggingface \
  veeksha-voxtral:latest \
  mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --host 0.0.0.0 --port 8025 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt '{"audio":1}' \
  --tokenizer-mode mistral --config-format mistral --load-format mistral \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}'
```
