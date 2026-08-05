Named benchmarks
================

Named benchmarks are versioned workload contracts built on top of ordinary
Veeksha benchmark runs.  A published benchmark ID is not edited in place;
material changes receive a new versioned ID.  The contract standardizes
*what* is run while keeping provider, model, endpoint, and deployment selection
separate.

Design and grouping
-------------------

The hierarchy is deliberately small:

``catalog collection``
    A dashboard or product grouping only.  It does not affect execution or
    scoring.

``benchmark``
    One product question with one interaction and scoring contract.  The
    benchmark pins input/output modes, pacing, audio format, latency clocks,
    execution settings, metrics, and dataset selections.

``dataset selection``
    One pinned dataset/config/split.  Every selection becomes its own normal
    Veeksha run and therefore has its own config, request rows, health report,
    and metric summary.

``target``
    A provider/model endpoint or a managed serving engine.  It is supplied at
    run time through an ordinary Veeksha target configuration, and is never
    embedded in a named benchmark.

Dataset selections belong in the same benchmark only when all of the following are
identical: modality, static/streaming input and output modes, pacing, codec,
connection and latency semantics, scorer, SLO contract, repetitions, and
metric definitions.  A telephony run and a full-band run, or complete-text TTS
and duplex text-streaming TTS, must use different benchmark IDs even when they
reuse prompts.

Dataset-level results are mandatory
-----------------------------------

For a benchmark with seven datasets and three targets, Veeksha executes 21
child runs.  It does not flatten the datasets into one trace.  The parent
result contains:

* every child's complete metric dictionary;
* explicitly resolved metrics for each dataset;
* request-level artifacts under the child's run directory;
* only those cross-dataset reductions declared by the manifest.

Latency percentiles are recomputed from pooled request observations.  P90s are
never averaged.  Corpus WER and CER are computed as the sum of edit counts
divided by the sum of reference units.  Equal-dataset macro means, when
desired, must be declared as separate metrics rather than silently replacing
corpus weighting.

Running a named benchmark
-------------------------

First create or choose a target configuration that satisfies the benchmark's
declared capability contract.  For the Indic ASR starter, the exact target must
declare either request-level language routing or provider auto-detection, plus
coverage for all 16 required language codes.  For example::

    client:
      type: stt
      provider: vajra_openai_realtime
      api_base: https://YOUR_ENDPOINT
      model: YOUR_MULTILINGUAL_ASR_MODEL
      sample_rate: 16000
      ws_chunk_size: 640
      ws_realtime_pacing: true
      language_mode: auto
      supported_languages: [bn, bho, hne, gu, hi, kn, mag, mai, ml, mr, or, pa, sa, ta, te, ur]

The declaration is an assertion about that exact provider/model/endpoint; the
runner validates the declaration but cannot manufacture unsupported language
coverage.  Then validate and materialize every child configuration without
contacting the endpoint::

    veeksha named-benchmark \
        --benchmark asr.indic.multidomain16.v1 \
        --target_config /path/to/verified-indic-asr-target.yml \
        --dataset_root /datasets/veeksha \
        --output_dir benchmark_output/named \
        --dry_run true

Set ``--dry_run false`` (or omit it) to execute.  A target config can use
``!expand`` to select several provider models; only its ``client``,
``endpoint``, and ``server`` fields are used.  Workload, traffic, evaluator,
and runtime fields come from the named benchmark.

The parent directory contains ``benchmark_manifest.json``, compiled child
configs, ``benchmark_summary.json``, ``dataset_results.jsonl``, failure rows,
and the ordinary Veeksha output tree for every dataset and target.  Partial
results are preserved if one child fails.

Indic starter benchmarks
------------------------

``asr.indic.multidomain16.v1``
    Seven independently reported configs from the pinned Indic ASR Eval
    compilation: Kathbath, Kathbath Noisy, FLEURS, IndicTTS, RESPIN, Common
    Voice, and GramVaani.  Audio is paced in real time at 16 kHz.  The scorer
    uses Unicode-preserving WER and CER and also records language and
    dataset-by-language summaries.  The source must first be materialized into
    WAV files and trace manifests under ``dataset_root``.

    Prepare the canonical full source in deterministic source order::

        uv run python scripts/prepare_hf_audio_trace.py \
            --output-dir /datasets/veeksha/asr.indic.multidomain16.v1

    ``--max-samples`` exists only for smoke testing and marks the prepared data
    noncanonical.  Audio and prompts remain outside the repository; the
    preparation metadata records the pinned source commit and manifest hash.

``tts.indic.robustness11.static-stream.v1``
    All 959 prompts in Sarvam's pinned 11-language robustness set, with exact
    text and language/category metadata preserved.  Complete text is sent and
    output audio is streamed.  This first version measures latency,
    continuity, stream-delivery RTF, and failures.  It requires a WebSocket
    target that returns mono 24 kHz PCM16, routes each prompt language (or uses
    provider-native auto-detection), and declares coverage for the benchmark's
    11 language codes.  It intentionally does not claim TTS intelligibility or
    naturalness quality until a pinned multilingual ASR judge and human
    evaluation protocol are integrated.

Metric clocks
-------------

The starter contracts intentionally separate connection setup from the
headline stream clocks.

For TTS, one request opens a WebSocket and waits for any provider session-ready
event.  ``trigger_to_first_playable_audio_ms`` starts when the client invokes
the provider-native synthesis-triggering message and stops when a complete 20
ms mono PCM16 frame has arrived.  For implicit-trigger APIs, the triggering
message is the complete-text append itself; for explicit-trigger APIs, it is
the separate synthesis command.  At 24 kHz PCM16 one frame is 960 decoded
audio bytes.  ``ttfc`` stops at the first non-empty decoded
audio payload and is diagnostic because a provider may fragment that payload
below one playable frame.  ``streaming_rtf`` covers first-to-last audio
delivery divided by playable audio delivered after the first payload, so it
measures steady-state delivery rather than startup.  ``ws_connect_latency_ms``
and connection-inclusive request RTF are reported separately.  Fluidity
replays the received PCM supply against 20 ms playback deadlines; 1.0 means no
observed underrun under the stated startup policy, not low TTFA or high speech
quality.

For ASR, audio is sent as 20 ms, 16 kHz mono PCM16 frames in realtime.
``time_to_first_visible_text`` starts when the first audio byte is sent and
stops at the first non-empty provider transcript event.  It is a stream-start
diagnostic, not end-of-word interactivity; this corpus does not provide the
reference word timestamps required for that headline metric.
``time_to_final_transcript`` starts at
audio EOF and stops at the provider's terminal transcript event.
``streaming_rtf`` covers first audio send to terminal transcript divided by
input duration, and connection-inclusive request RTF plus WebSocket setup are
diagnostics.  Interim transcript behavior remains provider-native and is not
used for WER; final WER/CER score the terminal transcript only.  WER/CER are
withheld unless every request completes successfully; error and cancellation
rates are reported separately, so unscored clips cannot make correctness look
artificially better.

Implementation boundary
-----------------------

The named layer is not another benchmark engine.  It adds a strict catalog
schema, deterministic compilation into existing ``BenchmarkConfig`` objects,
dataset-aware aggregation, and provenance.  Provider adapters, scheduling,
request execution, raw metrics, SLO evaluation, and server orchestration remain
in Veeksha's existing paths.

The next production steps after these starters are full endpoint smoke runs,
slice-level TTS reports, multilingual TTS quality scoring, and then a result
store/scheduler for the portal.  The portal should consume the artifacts above
rather than introducing a second result format.
