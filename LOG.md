# Universal Voice Reconciliation Log

## Status and scope

This branch is the feature-complete reconciliation of:

- `users/anirudha/tts_v2` at `60a181f9279410b23eab1a7ced0ae9131c349c1e`
- `asmitks:users/asmitks/tts-duplex-fluidity` at `94a015cc7acd07c68185a7ccde244ebebf2dc604`, reviewed as [project-vajra/veeksha#203](https://github.com/project-vajra/veeksha/pull/203)
- destination: `users/ksukrit/universal_voice`

Reconciliation date: 2026-07-21.

Both source tips have the exact same parent baseline,
`users/ksukrit/tts_merge` at
`44ada94297c00219f691c5921360c3bac32566ea`. Therefore the comparison is
not between two drifting repositories: it is two independent feature series
built on the same ASR/TTS baseline.

The destination was built from the `tts_v2` tip and merged with the PR 203
tip using a two-parent merge. Source commits and attribution remain in history;
nothing was squashed.

| Source series | Commits after common base | Diff from common base |
| --- | --- | --- |
| `users/anirudha/tts_v2` | 4 | 20 files, +3,780 / -84 |
| PR 203 / `users/asmitks/tts-duplex-fluidity` | 2 | 34 files, +3,263 / -45 |

### Source commit inventory

`tts_v2`:

1. `b863ca9f` - native Vajra streaming-text WebSocket client
2. `d4bb354e` - silent duration-at-cap truncation detection
3. `34023715` - zombie-session health check using Vajra worker telemetry
4. `60a181f9` - position-resolved long-form TTS scoring CLI

PR 203:

1. `183e7c56` - duplex and audio-fluidity benchmarking
2. `94a015cc` - validated streaming-latency and quality metrics

## Meaning of verification in this log

Each feature is classified at one of three levels:

- **Code verified**: configuration, registration, implementation, and data flow
  were inspected in the reconciled tree.
- **Unit verified**: the relevant deterministic tests passed in this tree.
- **Operational validation required**: a real provider, Vajra deployment,
  model checkpoint, GPU, or corpus is required. Those checks cannot be honestly
  replaced by mocks and are called out explicitly.

No claim of live-provider or live-server validation is made by this log.

## Features already present in both branches

These capabilities are identical because they are inherited from the exact
common baseline, not because one branch reimplemented the other.

| Common capability | What is present |
| --- | --- |
| Audio task routing | Explicit `AudioTask.TTS`, `AudioTask.STT`, and `AudioTask.LLM_AUDIO` routing through the audio evaluator |
| Canonical HTTP TTS | OpenAI-compatible `/v1/audio/speech` client with audio response capture |
| OpenAI realtime TTS | WebSocket text-in/audio-out client with the legacy `complete_text` response mode |
| Deterministic text pacing | `TextPacingConfig`, exact text segmentation, fixed or Poisson gap distribution, initial delay, seeded pacing |
| Normalized audio contract | Raw PCM metadata, sample rate, chunk timestamps, text-delta timestamps, commit/done offsets, and audio artifacts |
| Playback analysis | Local replay of received audio timing under startup/buffering policies |
| Baseline TTS quality | Seed-TTS-style WER and UTMOS request verification |
| Realtime ASR clients | Vajra OpenAI-realtime and vLLM-realtime PCM16 WebSocket clients behind one `STTClient` factory |
| ASR workload preparation | Audio trace flavor, manifest metadata, expected transcripts, word timestamps, and concurrency sweep support |
| ASR evaluation | Final/partial transcripts, WER, RTF, first-transcript latency, transcript snapshots, and word-visibility interactivity |
| Benchmark plumbing | Client registry, Vidhi config deserialization, result summaries, plots/artifacts, managed Vajra endpoints, and sweep orchestration |

## Features only in `users/anirudha/tts_v2`

All of these capabilities are included in the destination.

| Feature | Implementation and evidence | Verification |
| --- | --- | --- |
| Native Vajra streaming-text TTS | `VajraTTSStreamClient` uses `/v1/audio/speech/stream`; sends `session.config`, paced `input.text` events, then `input.done`; receives binary PCM plus `audio.start`, `audio.done`, and `session.done` | Code and unit verified |
| Silent server-cap truncation detection | `AudioChannelPerformanceConfig.max_expected_audio_ms` validates a positive cap; requests within 320 ms of the cap are marked `suspected_length_cap_truncation`; health fails when marked requests exist | Code and unit verified |
| Vajra zombie-session detection | Start/end snapshots of `/debug/tts_worker_stats` compare talker finished-session deltas with benchmark completions; surplus fails, deficit is noted, missing/unavailable telemetry skips | Code and unit verified; live Vajra telemetry required |
| Long-form TTS scoring CLI | `veeksha score-tts-longform` accepts WAV or raw PCM and writes `summary.json`, `curves.csv`, and `report.txt` | Code and unit verified |
| Position-resolved long-form WER | At most 28-second Whisper chunks, one global jiwer alignment, operation attribution back to time spans, and duration-weighted 60-second buckets | Code and unit verified; model/corpus run required |
| Long-form perceptual/identity tracks | 10-second UTMOS windows and optional 3-second WavLM speaker-similarity windows | Code and unit verified; checkpoints/GPU may be required |
| Long-form failure diagnostics | Repeated n-grams, compression-ratio loop signal, RMS/silence tracks, chunk and bucket curves | Code and unit verified |
| Shared WebSocket error handling | Reusable nested-`ExceptionGroup` flattening and transport-error priority in `veeksha.client.utils` | Code and unit verified |
| Seed-exact TTS normalization | Apostrophes retained, punctuation stripped, one double-space replacement pass, lowercase, and no digit-to-word normalization | Code and unit verified |
| Reusable UTMOS loader/scorer | Cached TorchScript loading and finite-score handling shared by request and long-form verification | Code and unit verified; checkpoint/GPU may be required |

## Features only in PR 203 / the duplex-fluidity branch

All of these capabilities are included in the destination.

| Feature | Implementation and evidence | Verification |
| --- | --- | --- |
| ElevenLabs streaming TTS | Native `/stream-input` protocol with paced incremental text, PCM output, provider auth, and normalized timeline metrics | Code and unit verified; live provider required |
| Deepgram Flux streaming TTS | Native `/v2/speak` protocol with paced text, concurrent send/receive, raw PCM, and normalized timeline metrics | Code and unit verified; live provider required |
| Deepgram Aura streaming TTS | Native `/v1/speak` adapter, including audio arriving after flush | Code and unit verified; live provider required |
| ElevenLabs HTTP TTS | Provider-native non-streaming HTTP request and complete PCM response capture | Code and unit verified; live provider required |
| Deepgram Flux HTTP TTS | Provider-native non-streaming HTTP request and complete PCM response capture | Code and unit verified; live provider required |
| OpenAI realtime duplex mode | `input_output_mode: duplex` can issue the response after `duplex_start_after_tokens` while input pacing continues; `complete_text` remains supported | Code and unit verified; live endpoint required |
| Provider identity contract | Request metrics carry provider, model, and protocol identifiers | Code and unit verified |
| Response-trigger timeline | Trigger and response-created offsets are distinct from text input, commit, first byte, audio done, and response done | Code and unit verified |
| First playable PCM metrics | Configurable frame duration (20 ms default) computes first-input-to, trigger-to, and request-start-to-first-playable audio | Code and unit verified |
| Etalon-inspired audio fluidity | Deadline-based fluidity index is invariant to transport fragmentation and carries early-frame slack forward | Code and unit verified |
| Stall simulation | Zero-delay, fixed-startup-delay, and buffer-target stall counts, total stall time, longest stall, and stall-free fraction | Code and unit verified |
| User/service attribution | User-observed fluidity is separate from TTS-service-attributable fluidity; duplex is fail-closed unless attribution is proven or `source_oversupplied` is selected | Code and unit verified |
| Duplex overlap | Records whether playable output overlaps ongoing text input and the overlap duration | Code and unit verified |
| Policy plots/artifacts | Raw timing data plus fluidity/stall summaries and stall-free-vs-startup plots | Code verified; plot generation exercised by broader suite |
| Fluidity-aware SLOs | Higher-is-better direction for fluidity/duplex success and lower-tail threshold evaluation | Code and unit verified |
| ASR empty-reference correctness | Empty reference + empty hypothesis is 0%; empty reference + non-empty hypothesis is 100% | Code and unit verified |
| ASR aggregate correctness | Per-sample mean, corpus edit-count WER, and duration-weighted WER are kept distinct | Code and unit verified |
| Strict TTS verification | Missing audio/transcription failures fail closed, Whisper language/task/beam settings are pinned and validated, and UTMOS failures do not bias the denominator | Code and unit verified |
| Dependency hygiene | Heavy `librosa` and `transformers` imports are lazy; Python 3.14 static-analysis configuration is included | Code verified |
| TTS user documentation | New TTS benchmarking guide and docs index wiring | Code verified |

## Replicated intent and how it was reconciled

The two series overlap heavily in purpose even where their concrete providers
or metrics differ. Keeping both implementations without a contract audit would
have produced clients that benchmark differently. These were reconciled
feature by feature.

| Replicated concern | `tts_v2` form | PR 203 form | Final decision |
| --- | --- | --- | --- |
| Paced incremental text | Vajra native streaming client | ElevenLabs, Deepgram, and OpenAI realtime duplex | Include every provider; use the same pacing primitives |
| Raw streaming timeline | Vajra text/audio/done offsets | Provider, trigger, playable-frame, fluidity fields | Include the union and make Vajra emit the provider/trigger contract too |
| WebSocket failure mapping | Shared recursive flattener with transport priority | Provider client-local flattener and detailed provider errors | Supersede only the duplicate flattener with the shared helper; retain provider-specific status/body mapping |
| Client registration/config | Vajra streaming enum/config/registry | Five provider-native clients plus Aura | Include all six additions with unique stable enum IDs |
| Audio evaluation | Duration-cap truncation | Provider labels, playable-frame timing, stalls/fluidity | Include both in the same request row and run summary |
| TTS WER normalization | Seed-exact normalization | Verification hardening and manual-oracle tests | Keep Seed-exact implementation and update the conflicting test oracle |
| UTMOS | Shared loader/scorer and long-form windows | Strict per-request accounting | Include both; one shared scoring primitive |
| Completion summaries | Realtime and Vajra streaming | New native streaming providers | Treat all text-in/audio-out streaming clients as realtime TTS summary producers |
| Test isolation | Optional models assumed absent in one environment | Lazy optional dependencies | Make absence deterministic with monkeypatches; never download a model during a unit test |

## Conflict and decision ledger

There were five textual conflicts. All were additive capability conflicts; none
required choosing one user-facing feature over another.

| File/area | Decision | Classification | Verification |
| --- | --- | --- | --- |
| `veeksha/client/realtime_tts.py` | Preserve PR duplex behavior and use the shared `tts_v2` WebSocket exception helper/imports | Included + superseded duplicate plumbing | Realtime duplex/complete-text tests |
| `veeksha/config/client.py` | Preserve Vajra plus all five provider-native configs and Aura streaming config | Included | Six config deserialization cases |
| `veeksha/config/evaluator.py` | Preserve both duration-cap validation and fluidity/stall policy validation | Included | Evaluator config/task tests |
| `veeksha/evaluator/performance/audio.py` | Preserve truncation marking together with provider labels, playable-frame, overlap, fluidity, stall, plot, and summary fields | Included | Audio task/interactivity/SLO tests |
| `veeksha/types/__init__.py` | Keep existing values 1-7; assign provider additions 8-12 to eliminate branch-local collisions | Included with reconciled IDs | Enum uniqueness/stability test |
| Three cleanly merged overlapping files | `registry.py`, audio evaluator tests, and `verification/audio.py` retained both sides and were audited after Git's clean merge | Included | Registry/config/verification tests |
| Native provider exception groups | Prefer fatal provider error over generic socket leaves while still using the shared recursive helper | Superseded duplicate plumbing | Nested `ExceptionGroup` test |
| Vajra provider/timing metadata | Add provider/model/protocol, first input trigger, and response-created timing to the native Vajra stream | Included semantic integration | Vajra fake-WebSocket contract test |
| Streaming summary classification | Add ElevenLabs streaming, Deepgram Flux streaming, and Deepgram Aura streaming to the realtime completion-summary set | Included semantic integration | Explicit membership test |
| Seed normalization test expectation | Replace PR's pre-reconciliation double-space expectation with the Seed-exact single replacement result for that fixture | Superseded test oracle, not capability | Manual WER/normalization tests |
| Optional-model unit test | Patch all optional model builders to unavailable instead of relying on the machine lacking packages/GPU | Superseded nondeterministic test setup | Long-form graceful-degradation test |
| Added-code static diagnostics | Use typed dynamic imports for optional models, compatible jiwer lookup, explicit ndarray conversion, robust health sampling, and a statically visible version fallback | Included integration hardening | Added files: 0 pyright errors; full tree improves baseline by 6 |

### Included, removed, and superseded summary

- **Included:** every user-facing feature from both source series.
- **Removed:** no user-facing feature, client, metric, health check, verifier,
  CLI, artifact, or documentation page.
- **Superseded implementation details only:** the provider-local exception
  flattener, conflicting numeric enum assignments, a non-Seed normalization
  expectation, and an environment-dependent unit-test assumption.

## Final client type mapping

The mapping intentionally preserves the common baseline and `tts_v2` values.

| ID | Client type | Source |
| ---: | --- | --- |
| 1 | `OPENAI_CHAT_COMPLETIONS` | common baseline |
| 2 | `OPENAI_COMPLETIONS` | common baseline |
| 3 | `OPENAI_ROUTER` | common baseline |
| 4 | `TTS` | common baseline |
| 5 | `REALTIME_TTS` | common baseline |
| 6 | `STT` | common baseline |
| 7 | `VAJRA_TTS_STREAM` | `tts_v2` |
| 8 | `ELEVENLABS_STREAMING_TTS` | PR 203 |
| 9 | `DEEPGRAM_FLUX_STREAMING_TTS` | PR 203 |
| 10 | `ELEVENLABS_HTTP_TTS` | PR 203 |
| 11 | `DEEPGRAM_FLUX_HTTP_TTS` | PR 203 |
| 12 | `DEEPGRAM_AURA_STREAMING_TTS` | PR 203 |

## Complete ASR pipeline inventory

### 1. Workload and input preparation

- `TraceFlavorType.AUDIO` generates request-scoped audio workloads.
- Manifest metadata carries audio paths, expected transcripts, duration, and
  optional reference word timestamps.
- PCM16 is streamed from the audio input; clip-level preparation is cached for
  repeated concurrent sessions.
- Input start/end timing is propagated for evaluator attribution.
- STT-specific concurrency sweep specs exist for Vajra and vLLM realtime
  providers.

### 2. Client protocols and streaming lifecycle

- `STTClientConfig.provider` selects `vajra_openai_realtime` or
  `vllm_realtime`.
- Both clients share one lifecycle: open session, encode PCM chunks, send EOF,
  and parse provider events.
- Send and receive loops run concurrently.
- Optional 1x realtime upload pacing prevents TTFC from being measured only
  after a full clip upload.
- Metrics include first transcript chunk, final completion, end-to-end latency,
  real-time factor, partial/final transcript, snapshots, audio byte/sample
  metadata, and task identity.

### 3. Correctness and aggregation

- The vendored Open ASR Leaderboard English normalizer is applied before jiwer
  alignment.
- Per request: substitutions + deletions + insertions, reference word count,
  partial WER when available, and final WER.
- Empty reference behavior is explicit: 0% for no hallucination and 100% for
  any non-empty hypothesis.
- Run summaries keep sample-mean WER, corpus edit-count WER, and
  duration-weighted WER separate for both final and partial results.

### 4. ASR interactivity

- Transcript snapshots are matched against normalized, timestamped reference
  words.
- Word visibility latency is measured from each reference word's end time to
  the first matching partial snapshot.
- Missing word timestamps or transcript snapshots make the metric unavailable
  rather than fabricating a value.
- Repeated partial transcripts reuse matching work without changing semantics.

### 5. ASR outputs and boundaries

- Request CSV rows and run summaries include correctness, timing, and
  interactivity fields.
- ASR work shares the audio evaluator with TTS but is explicitly routed through
  `AudioTask.STT`; TTS-only playback and truncation logic is not applied to
  ASR.
- The new branch-specific ASR change is correctness hardening from PR 203;
  the rest of the ASR pipeline comes unchanged from the common baseline.

## Complete TTS pipeline inventory

### 1. Workload and text pacing

- Seed-TTS text trace support and generic text-channel requests.
- Exact deterministic segmentation that reconstructs source text.
- Configurable tokens/second, tokens/delta, fixed or Poisson gaps, initial
  delay, and seed.
- Text-delta timestamps are retained for input/output overlap and latency
  attribution.

### 2. Supported TTS clients

| Public client | Transport/mode | Input/output behavior |
| --- | --- | --- |
| `tts` | OpenAI-compatible HTTP | Complete text -> complete audio |
| `realtime_tts` | OpenAI realtime WebSocket | Paced text -> audio; `complete_text` or early-trigger `duplex` |
| `vajra_tts_stream` | Vajra native WebSocket | `session.config` + paced text + done -> binary PCM/events |
| `elevenlabs_streaming_tts` | ElevenLabs native WebSocket | Paced text -> streaming PCM |
| `deepgram_flux_streaming_tts` | Deepgram Flux native WebSocket | Paced text -> streaming PCM |
| `deepgram_aura_streaming_tts` | Deepgram Aura native WebSocket | Paced text/flush -> streaming PCM |
| `elevenlabs_http_tts` | ElevenLabs native HTTP | Complete text -> complete PCM |
| `deepgram_flux_http_tts` | Deepgram Flux native HTTP | Complete text -> complete PCM |

### 3. Normalized response contract

- Provider, model, and protocol.
- Raw-PCM flag, sample rate, audio bytes, chunk count, and per-chunk
  arrival/byte timestamps.
- Text delta timestamps, response trigger, response created, input commit,
  first audio byte/chunk, audio done, and response/session done.
- Input text and `AudioTask.TTS` identity.
- Provider-specific errors retain HTTP/WebSocket status/body detail while
  nested task-group failures resolve to the most informative leaf.

### 4. Performance and interactivity

- TTFC and end-to-end latency.
- First playable frame is computed from accumulated PCM duration, independent
  of provider transport fragmentation.
- First-input-to-first-playable, trigger-to-first-playable, and
  request-start-to-first-playable latency.
- Duplex overlap observed and overlap duration.
- Etalon-inspired fluidity deadlines with configurable frame and startup delay.
- User-observed versus service-attributable fluidity.
- Zero-delay, fixed-delay, and buffer-target playback simulations.
- Stall count, total/longest stall, stall-free flags/fractions, and policy
  curves.
- Silent length-cap truncation suspicion for configured expected caps.

### 5. Quality and verification

- Request-level WER using Seed-TTS normalization.
- Pinned faster-whisper language/task/beam settings.
- Strict fail-closed handling for missing audio, failed transcription, and
  threshold violations.
- UTMOS with finite-score validation and unbiased failure accounting.
- Long-form WER, UTMOS, optional speaker similarity, repetition,
  compression-ratio, RMS, silence, time buckets, and report artifacts.

### 6. Health, SLOs, and artifacts

- Health fails on suspected silent length-cap truncation.
- Optional Vajra zombie-session probe compares worker telemetry snapshots.
- SLO direction supports lower-is-better latency/stall/error metrics and
  higher-is-better fluidity/duplex success.
- Request CSVs, raw timing, summaries, WAV/audio artifacts, fluidity/stall
  plots, long-form JSON/CSV/text reports, and TTS benchmarking documentation.

## Verification ledger

| Check | Result |
| --- | --- |
| Exact common ancestor and source SHAs | PASS |
| Two-parent merge topology retained | PASS: both source tips are parents of the reconciliation merge commit |
| Unresolved conflict markers | PASS |
| `git diff --check` | PASS |
| Python compile check for reconciliation edits | PASS |
| Every unit-test file changed by either source branch | PASS: 135 passed, 0 failed, 0 skipped |
| Full repository unit suite | PASS with host caveat: 428 passed and 2 unchanged port-allocation tests deselected; unfiltered run was 428/430 with only port 8000 occupancy failures |
| Formatter/import checks for changed Python | PASS: Black and isort across all Python changed from the common base |
| Static analysis | PASS for regression: 62 errors vs 68 on the exact common baseline, with no diagnostic unique to this branch; inherited repository-wide debt remains |
| Live ElevenLabs/Deepgram/OpenAI provider calls | NOT RUN: credentials and billable external services required |
| Live Vajra native streaming + telemetry probe | NOT RUN: deployed endpoint and telemetry directory required |
| Long-form real-corpus/model run | NOT RUN: audio/reference corpus and model checkpoints required |

## Operational requirements and known boundaries

- ElevenLabs clients require `ELEVENLABS_API_KEY`; Deepgram clients require
  `DEEPGRAM_API_KEY`. Live calls can incur provider cost.
- Realtime OpenAI/Vajra clients require a compatible deployed endpoint and any
  configured bearer token.
- The zombie probe is enabled only for a Vajra endpoint with a health URL and
  `VAJRA_TTS_TELEMETRY_DIR`; 404/unavailable telemetry is reported as a skip.
- Duration-cap truncation detection only runs when
  `max_expected_audio_ms` is explicitly configured.
- UTMOS, Whisper, and WavLM tracks degrade with explicit notes when dependencies
  or checkpoints are unavailable. Unit tests never fetch a model.
- Service-attributed fluidity is intentionally conservative during duplex
  input. Use `source_oversupplied` only when the source is known to have
  supplied enough audio/text to attribute misses to the TTS service.
- Provider protocol adapters are covered with deterministic fake
  HTTP/WebSocket tests. A release using external providers should still run a
  credentialed smoke test against each selected provider/model.
