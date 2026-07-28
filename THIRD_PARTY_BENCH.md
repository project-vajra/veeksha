# Third-party ASR / TTS benchmarking

Summary of the hosted-provider voice experiments on `users/ksukrit/universal_voice`:
what was run, which knobs were set and why, the numbers we have, and the caveats
that apply before anything is published.

All runs use one Veeksha client, one trace, one metric contract per modality; only
the wire adapter (`client.provider`) changes across vendors.

---

## 1. ASR (streaming STT)

### Dataset

`traces/asr/aa_public/manifest.jsonl` — 100 clips, built by
`scripts/prepare_audio_traces.py` from the Artificial Analysis cleaned mirrors:

| Split | Clips | Total audio | Duration p50 (min–max) |
|---|---|---|---|
| `aa_voxpopuli` (VoxPopuli-Cleaned-AA, test) | 50 | 499 s | 8.4 s (5.0–23.0) |
| `aa_earnings22` (Earnings22-Cleaned-AA) | 50 | 1350 s | 29.8 s (11.4–30.0) |

4,429 reference words total; all 100 rows carry `reference_word_timestamps` from
NeMo forced alignment (`nvcr.io/nvidia/nemo:26.02`, run in Docker at trace-prep
time). `--max-duration 30` splits long Earnings22 sources on word boundaries.

Rationale: two public splits give an externally recognizable reference point —
VoxPopuli is clean read/parliamentary speech, Earnings22 is harder accented
conversational speech, so the pair separates "easy" from "real" WER. This is
deliberately *not* an AA reproduction: AA-AgentTalk is proprietary and AA's
normalizer is not open source, so we use the Open ASR Leaderboard English
normalizer and report corpus WER as primary.

### Veeksha knobs (`veeksha/sample_configs/stt_*.yml`)

| Knob | Value | Why |
|---|---|---|
| `client.type` | `stt` | One provider-agnostic realtime WS client; `provider` only swaps the wire adapter. |
| `ws_realtime_pacing` | `true` | Audio is fed at 1× wall clock. Without it, `interactivity` is meaningless. |
| `ws_chunk_size` | 2560 B (80 ms) — 3200 B (100 ms) for Cartesia | Matches each vendor's recommended input cadence; PCM16 mono @16 kHz. |
| `sample_rate` | 16000 | Required by every hosted endpoint (hard-validated for Together). |
| `traffic_scheduler` | `concurrent`, `target_concurrent_sessions: 1` | Hosted APIs: we measure provider steady-state latency/quality, not our capacity. Their batching is not ours to control. |
| `runtime.max_sessions` | 100 | One pass over the trace, sequential. |
| `seed` | 42 | Same trace order for every provider. |
| SLO | P90 `interactivity` < 1000 ms | The brief's realtime bar. |

### Headline metric

`interactivity`, not `ttfc`: for every reference word matched into the evolving
transcript, `first snapshot containing the word − reference end-of-word time`;
the request value is the mean over matched words. It measures perceived
transcription lag at spoken-word boundaries. `ttfc` /
`time_to_first_visible_text` are kept as diagnostics — they reward providers that
emit an early scrap of text, which is not the same thing.

### Results — 100 sequential requests each

Source: `benchmark_output/asr_100_sequential_summary.{csv,json}` (runs 23 Jul 2026),
plus the two later runs below.

| Provider / model | Corpus WER % | VoxPopuli | Earnings22 | Interactivity p50 / p90 (ms) | First visible text p50 (ms) | RTF p50 | SLO |
|---|---|---|---|---|---|---|---|
| Deepgram Flux `flux-general-en` | 8.96 | 4.61 | 10.76 | 169 / 226 | 727 | 1.02 | pass |
| Deepgram Nova-3 `nova-3` | 10.99 | 4.14 | 13.82 | 465 / 545 | 1032 | 1.04 | pass |
| ElevenLabs `scribe_v2_realtime` | 6.25 | 2.19 | 7.92 | 740 / 858 | 2012 | 1.01 | pass |
| Cartesia `ink-2` *(stale, see below)* | 10.33 | 2.89 | 13.40 | 1019 / 1107 | 1166 | 1.02 | **fail** |
| Cartesia `ink-2` *(post-fix, 25 Jul, 96/97 req)* | **6.17** | 1.85 | 7.97 | 1026 / 1111 | 1346 | 1.02 | **fail** |
| Mistral `voxtral-mini-transcribe-realtime-2602` | 7.75 | 2.34 | 9.98 | 830 / 906 | 1258 | 1.13 | pass |
| Together `nemotron-3-asr-streaming-0.6b` (26 Jul, 100 req) | 14.43 | 6.32 | 17.78 | 320 / 410 | — | — | pass |
| Together `nemotron-3.5-asr-streaming-0.6b` (26 Jul, 94/95 req) | 18.76 | 8.48 | 23.09 | 403 / 514 | — | — | pass |

Read: Flux is the latency leader (226 ms P90) at mid-pack accuracy; Scribe v2 and
post-fix Ink-2 lead on accuracy but sit near or past the 1 s bar; Ink-2 is the only
model that fails the interactivity SLO. Together's Nemotron models are fast but
clearly the weakest on WER, and 3.5 is worse than 3 on this trace.

### ASR caveats

- **The Cartesia row in the summary CSV is stale.** Commit `3ac42a3` (25 Jul) fixed
  transcript assembly for Cartesia — it emits *incremental deltas*, not snapshots,
  so leading/trailing whitespace has to survive concatenation. Before the fix words
  were glued together: corpus WER 10.33 % → **6.17 %** on the rerun. The summary CSV
  was generated before that and has not been regenerated.
- The Together Nemotron 3.5 run is partial (95 of 100 dispatched, 94 completed) and
  the two Together runs are not in the summary CSV.
- Every percentile in `summary_stats.json` is a **DDSketch estimate**
  (`relative_accuracy=0.001`), not an exact order statistic. Two different runs can
  report a byte-identical percentile because both true values land in the same
  logarithmic bucket. Recompute from `request_level_metrics.jsonl` for anything
  precision-sensitive.

---

## 2. TTS (streaming, WebSocket)

### Matrix

5 models × 2 text corpora × 100 requests, concurrency 1, 23 Jul 2026 —
`third_party_provider_main_100_20260723/` (an earlier 50-request pass sits in
`third_party_numbers/` and is superseded).

Corpora — chosen to bracket the two real shapes of TTS input:

| Corpus | Source | Chars p50 (min–max) | Words p50 | Audio p50 | Role |
|---|---|---|---|---|---|
| SeedTTS | `TwinkStart/Seed-TTS-Eval`, `en`, train | 57 (20–121) | 10 | ~3.2 s | Short prompts; a published, citable eval set. |
| ShareGPT | local ShareGPT JSON, `gpt` turns | 245 (20–497) | 40 | ~15 s | Long assistant turns — the realistic voice-agent load. |

### Veeksha knobs (`veeksha/sample_configs/tts_streaming_*.yml`)

| Knob | Value | Why |
|---|---|---|
| `client.type` | `streaming_tts` | Text is paced in over a live WS session, not posted whole. |
| `pacing` | 50 whitespace words/s, 1 word per delta, `fixed` gap, `initial_delay_s: 0` | Far faster than any LLM emits, so the provider is never text-starved. Isolates TTS from upstream. Recorded per request as `text_pacing_unit=whitespace_word`. |
| `sample_rate` | 24000 | Common PCM16 output format across all five models. |
| `fluidity_frame_ms` / `startup_delay_ms_values` | 20 / `[0, 100, 300]` | 20 ms playback frame; primary score assumes zero artificial buffering, the other two are policy variants. |
| `fluidity_attribution_mode` | `conservative` | Report the observed user timeline; blame TTS only when all text arrived before playback. |
| `traffic_scheduler` | concurrency 1, `cancel_session_on_failure: true` | Steady-state per-request latency; fail loudly instead of retrying errors away. |
| `runtime.*_threads` | 1 each, `pregenerate_sessions: false` | Removes client-side scheduling jitter from a single-stream latency measurement. |
| Provider parity | ElevenLabs: Adam `pNInz6obpgDQGcFmaJgB`, `stability 0.5`, `similarity_boost 0.8`, `auto_mode: false`, `chunk_length_schedule [120,160,250,290]`, `apply_text_normalization: off`; Cartesia: Skylar `db6b0ed5…`, `cartesia_version 2026-03-01`, `max_buffer_delay_ms 3000` | Voice and buffering pinned so the two ElevenLabs models differ only by model; normalization off because it is plan-dependent. |

SLOs applied to every run: P90 < 1 s on `first_input_to_first_audio_ms`,
`trigger_to_first_playable_audio_ms` and `ttfc`; P90 `rtf` < 1; P1
`user_audio_fluidity_index` ≥ 0.99.

### Headline metric

`trigger_to_first_playable_audio_ms` — synthesis trigger (first
synthesis-eligible text append, *not* WS connect) to the first **complete 20 ms
playback frame** (960 B at 24 kHz PCM16). Stricter than "first non-empty wire
payload": a 30-byte teaser chunk is not audio a user can hear. Connection setup
is reported separately via `request_start_to_first_playable_audio_ms`.

### Results

`third_party_provider_streaming_tts_results.csv` (+ `_network_free.csv`).
TTFA = `first_input_to_first_audio_ms`; NF = network-free estimate.

| Model | Corpus | TTFA p50 / p90 (ms) | NF p50 (ms) | E2E p90 (ms) | RTF p90 | Fluidity P1 | SLOs |
|---|---|---|---|---|---|---|---|
| Deepgram `aura-2-thalia-en` | SeedTTS | 221 / 318 | 157 | 2948 | 0.75 | 1.00 | pass |
| Deepgram `flux-haley-en` | SeedTTS | 299 / 400 | 236 | 3587 | 0.84 | 1.00 | pass |
| ElevenLabs `eleven_flash_v2_5` | SeedTTS | 296 / 410 | 269 | 615 | 0.23 | 1.00 | pass |
| Cartesia `sonic-3.5` | SeedTTS | 305 / 416 | 231 | 1605 | 0.53 | 1.00 | pass |
| ElevenLabs `eleven_multilingual_v2` | SeedTTS | 553 / 746 | 526 | 964 | 0.29 | 1.00 | pass |
| Deepgram `aura-2-thalia-en` | ShareGPT | 266 / 440 | 202 | 13400 | 0.66 | 0.988 | **fail** (fluidity) |
| ElevenLabs `eleven_flash_v2_5` | ShareGPT | 487 / 570 | 460 | 1941 | 0.13 | 1.00 | pass |
| ElevenLabs `eleven_multilingual_v2` | ShareGPT | 949 / 1059 | 922 | 3220 | 0.19 | 1.00 | **fail** (P90 TTFB > 1 s) |
| Deepgram `flux-haley-en` | ShareGPT | 465 / 687 | 402 | — | — | — | **incomplete**: HTTP 408 at 41/100, reproduced on rerun |
| Cartesia `sonic-3.5` | ShareGPT | 477 / 797 | 403 | — | — | — | **incomplete**: HTTP 500 "invalid input" at 47/100, reproduced on rerun |

Read: every model clears 1 s TTFA on short SeedTTS prompts. Long ShareGPT turns are
where they separate — Aura-2 is fastest to first audio but the only model that
stalls during playback, Flash v2.5 is the strongest overall (570 ms P90, 0.13 RTF,
no stalls), and Multilingual v2 misses the 1 s bar. Two provider/corpus cells could
not be completed at all; the failures reproduced on rerun and are reported rather
than retried away.

### Network-free adjustment

`scripts/create_network_free_tts_results.py` opens N (=5) fresh authenticated WS
connections per endpoint, sends a ping with no synthesis text, and subtracts the
median ping/pong RTT from the observed TTFA percentiles. Measured RTTs were
~63 ms (Deepgram), ~74 ms (Cartesia), ~27 ms (ElevenLabs) from this host.

This is an estimate, not a decomposition: probes are unpaired and were taken
*after* the benchmark (24 Jul). For the two incomplete rows the observed
percentiles are reconstructed from the run's saved CDF over completed requests.
Use it to argue "the gap is not just our network", not as a compute-only number.

### TTS caveats

- **No quality scoring.** `verification.wer` and `utmos` are `enabled: false` in
  every config. 929 WAVs (453 MB) are saved under
  `third_party_provider_main_100_20260723/<model>/<corpus>/audio_quality/` for
  offline scoring later. Latency claims are currently unpaired with any
  intelligibility check.
- The `SESSION CONCURRENCY CHECK` failure recorded in every row is a **framework
  bug, not a provider result**: `health.py` added second-based dispatch timestamps
  to millisecond `end_to_end_latency`. Fixed in `aee4d9c8`, which landed ~3 h after
  these runs, so the artifacts still carry the false failure.
- DDSketch percentile caveat applies here too.
- The ShareGPT trace path in the configs is machine-local
  (`/scratch/sukrit/voice_eval/...`); the dataset is not in the repo.

---

## 3. Code changes on this branch

Feature work (committed):

- `fa2518bc`, `f56f9cac`, `4ab6f749` — hosted streaming adapters for ElevenLabs,
  Deepgram Flux/Aura, Mistral, Cartesia; voices pinned after smoke tests.
- `a6dd61ff` — the 100-request TTS benchmark matrix, its configs and test coverage.
- `f0d19a51` — headline latency aligned with the brief: TTS TTFB =
  `trigger_to_first_playable_audio_ms`, STT TTFB = `interactivity`.
- `47702507` — WS ping-RTT probe + network-free results script.
- `aee4d9c8` — health-checker unit fix (see caveat above).
- `3ac42a3b` — Cartesia STT delta assembly (`partial_delta` / `final_delta` kinds),
  the WER fix described above.

Uncommitted in the working tree:

- `veeksha/client/stt.py`, `veeksha/config/client.py`, docs, tests — the **Together
  realtime STT adapter** (`together_openai_v1_realtime_transcription`): OpenAI-shape
  realtime WS with `turn_detection=none` so endpointing is client-driven and the
  single completion event pairs with our explicit EOF commit; `sample_rate` pinned
  to 16 kHz.
- `veeksha/sample_configs/stt_together_nemotron{,_3_80ms}.yml`.
- `scripts/create_network_free_tts_results.py`.
- `.gitignore`: `/third_party_numbers/` → `/third_party_*/` (keeps result trees out
  of git).
- `vajra_c{4,32,128,144,156}*_epinto6.yaml` — in-house Vajra STT concurrency sweeps
  on the *same* `aa_public` trace, so these third-party numbers are the reference
  line for them. Note they use `ws_chunk_size: 8224` (~257 ms) rather than the 80 ms
  hosted cadence — a client-CPU tradeoff at c156, not an apples-to-apples input rate.

---

## 4. Reproducing

```console
# ASR (one config per provider)
uvx -p 3.14t veeksha benchmark --config veeksha/sample_configs/stt_deepgram_flux.yml

# TTS (one config per model × corpus)
uvx -p 3.14t veeksha benchmark \
  --config veeksha/sample_configs/tts_streaming_elevenlabs_flash_v2_5_sharegpt.yml

# Regenerate the trace
.venv/bin/python scripts/prepare_audio_traces.py --clips-per-dataset 128 --max-duration 30

# Aggregate + network-free adjust the TTS matrix
python third_party_provider_main_100_20260723/aggregate_results.py
python scripts/create_network_free_tts_results.py \
  third_party_provider_main_100_20260723/third_party_provider_streaming_tts_results.csv \
  --samples 5
```

API keys come from the env vars named by each config's `api_key_env`.

---

## 5. Before publishing

1. Regenerate the ASR summary with the post-fix Cartesia run and the Together rows.
2. Rerun the two failed ShareGPT TTS cells, or publish them explicitly as provider
   errors with the reproduction evidence.
3. Score the saved TTS WAVs (WER + UTMOS) with one pinned judge configuration
   across all providers — latency without an intelligibility check is not a result.
4. Rerun the TTS matrix (or re-derive health status) post-`aee4d9c8` so artifacts
   stop carrying the false concurrency-check failure.
5. Record region, pricing date and retry policy alongside any published table.
