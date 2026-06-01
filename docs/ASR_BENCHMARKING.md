# ASR Benchmarking

## Trace Generation

Generate the public ASR trace with:

```bash
python scripts/prepare_audio_traces.py --clips-per-dataset 128
```

This writes `traces/asr/aa_public/manifest.jsonl` plus WAV files under
`traces/asr/aa_public/audio/`.

The trace uses the public Artificial Analysis cleaned datasets, VoxPopuli and
Earnings22, as a recognizable external point of reference. It does not attempt
to reproduce the full Artificial Analysis benchmark exactly: the proprietary
AA-AgentTalk dataset is not available, their custom normalizer is not open
sourced, and Veeksha measures behavior across engines so any point of difference
is fine as long as it's consistent for all engines.

## Request Metrics

Request-level metrics are written to `request_level_metrics.jsonl`.

Common audio metrics:

- `ttfc`: time from client request start to the first transcript delta/final
  observed by the client, in milliseconds.
- `end_to_end_latency`: time from client request start to request completion, in
  milliseconds.
- `generated_audio_duration`: audio duration represented by the request, in
  milliseconds. For STT this is the input clip duration.
- `rtf`: real-time factor, computed as
  `end_to_end_latency / generated_audio_duration`.
- `chunk_count`: number of transcript deltas observed, or `1` when only a final
  transcript is returned.
- `input_tokens`: whitespace token count of the final transcript.

ASR-specific metrics:

- `time_to_first_partial`: time from the client sending EOF/commit for the audio
  stream to the first non-empty partial transcript received after that point, in
  milliseconds. This is only present when the provider emits such a partial.
- `time_to_final_transcript`: time from the client sending EOF/commit for the
  audio stream to the final transcript, in milliseconds.
- `partial_transcript`: first non-empty post-EOF partial transcript used for
  partial WER.
- `final_transcript`: final transcript returned by the provider.
- `expected_transcript`: reference transcript from the trace row.
- `partial_wer`: WER for `partial_transcript` against `expected_transcript`.
  Present for clip-scoped rows when a partial transcript is available.
- `final_wer`: WER for `final_transcript` against `expected_transcript`.
  Present for clip-scoped rows. For Earnings22 parent-scoped rows, final WER is
  computed after regrouping chunks, so per-request `final_wer` is left null.

## Partials and Finals

Streaming providers may emit many transcript deltas before the audio stream is
committed. Veeksha concatenates those deltas to track `ttfc` and the eventual
transcript, but `partial_transcript` is specifically the first non-empty
transcript observed after EOF/commit. `final_transcript` is the provider's final
message when available, otherwise the concatenated deltas are used as a fallback.

## Aggregate Metrics

Aggregate metrics are emitted in the performance summary.

General audio aggregates are percentile summaries over request-level values:

- `ttfc (Mean/P50/P90/P99)`
- `end_to_end_latency (Mean/P50/P90/P99)`
- `generated_audio_duration (Mean/P50/P90/P99)`
- `rtf (Mean/P50/P90/P99)`
- `chunk_count (Mean/P50/P90/P99)`
- `time_to_first_partial (Mean/P50/P90/P99)`, for STT runs when available
- `time_to_final_transcript (Mean/P50/P90/P99)`, for STT runs when available

ASR WER aggregates:

- `asr_final_sample_count`: number of scored ASR samples after parent regrouping.
- `asr_final_sample_mean_wer`: unweighted mean WER across scored samples.
- `asr_final_corpus_wer`: corpus-level WER from summed edit counts and summed
  reference words.
- `asr_final_duration_weighted_wer`: WER weighted by audio duration.
- `asr_partial_*`: same aggregate modes for partial transcripts, over samples
  where partial WER is available.
- `asr_dataset_<dataset>_final_*` and `asr_dataset_<dataset>_partial_*`:
  dataset-specific versions of the same metrics.

For comparisons between serving engines, prefer the same aggregate metric across
runs, with `asr_final_duration_weighted_wer` as the closest analogue to the
duration-weighted WER used by public ASR leaderboards.

## Manifest Fields

Each manifest row represents one audio request. Key fields:

- `session_id`: trace session identifier.
- `audio_file`: WAV path relative to the manifest directory.
- `dataset`: source dataset key, such as `aa_voxpopuli` or `aa_earnings22`.
- `expected_transcript`: reference transcript used for WER.
- `duration_s`: clip duration in seconds.
- `reference_scope`: `clip` for independently scored clips, `parent` for
  Earnings22 chunks that must be regrouped before WER is computed.
- `sample_id`: unique row/chunk identifier.
- `parent_id`: original Earnings22 call identifier for parent-scoped rows.
- `chunk_index`: zero-based chunk index within the parent call.
- `parent_num_chunks`: expected number of chunks for the complete parent call.

Earnings22 rows deliberately repeat the full parent `expected_transcript` on
each chunk. The evaluator uses `parent_id`, `chunk_index`, and
`parent_num_chunks` to concatenate chunk transcripts back into the original call
before computing parent-level WER. This also supports wrapped traces: if the
same parent appears multiple times in one run, the evaluator groups the first
occurrence of every chunk together, then the second occurrence, and so on.

## References

- Artificial Analysis Speech to Text Methodology:
  https://artificialanalysis.ai/speech-to-text/methodology
- VoxPopuli-Cleaned-AA:
  https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA
- Earnings22-Cleaned-AA:
  https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA
