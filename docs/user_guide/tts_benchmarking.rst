Streaming TTS benchmarks
========================

Veeksha measures TTS from the client: it paces text into the provider, records
when decoded PCM becomes playable, and derives every latency and continuity
metric from one monotonic request timeline. Provider clocks are not used.

Benchmark structure
-------------------

The TTS path has five layers:

1. A trace flavor turns source text into one-request TTS sessions.
2. The traffic scheduler controls arrival rate or concurrent sessions.
3. A TTS client implements the provider protocol and records text and PCM events.
4. The audio performance evaluator derives latency, RTF, overlap, stalls, and
   fluidity from those events.
5. The audio quality evaluator can save WAV files and compute WER and UTMOS.

The provider adapters intentionally retain their native lifecycle:

.. list-table::
   :header-rows: 1

   * - ``client.type``
     - Transport
     - Input behavior
   * - ``realtime_tts``
     - OpenAI Realtime-compatible WebSocket
     - Complete-text or duplex ``response.create`` scheduling
   * - ``elevenlabs_streaming_tts``
     - ElevenLabs ``/stream-input`` WebSocket
     - Paced partial text followed by the provider finalization message
   * - ``deepgram_flux_streaming_tts``
     - Deepgram Flux ``/v2/speak`` WebSocket
     - Paced ``Speak`` messages; audio may arrive before ``Flush``
   * - ``deepgram_aura_streaming_tts``
     - Deepgram Aura ``/v1/speak`` WebSocket
     - Paced ``Speak`` messages followed by ``Flush``
   * - ``elevenlabs_http_tts``
     - ElevenLabs complete-response HTTP
     - All text and then all audio
   * - ``deepgram_flux_http_tts``
     - Deepgram Flux ``POST /v2/speak``
     - All text and then all audio

The native protocols follow the vendor specifications for `ElevenLabs
stream-input <https://elevenlabs.io/docs/api-reference/text-to-speech/v-1-text-to-speech-voice-id-stream-input>`_,
`Deepgram Flux <https://developers.deepgram.com/docs/flux-tts/quickstart>`_,
and `Deepgram Aura <https://developers.deepgram.com/reference/text-to-speech/speak-streaming>`_.

Metrics
-------

.. list-table::
   :header-rows: 1

   * - Metric
     - Meaning
   * - ``request_start_to_first_playable_audio_ms``
     - Request start to the first complete PCM playback frame. This is the
       primary first-audio latency.
   * - ``ttfc``
     - Request start to the first wire audio chunk. A tiny partial chunk may not
       yet be playable, so this is retained as a transport diagnostic.
   * - ``rtf``
     - End-to-end request time divided by generated audio duration. It includes
       paced upstream text input.
   * - ``streaming_rtf``
     - Wall time from the first to last audio arrival divided by audio delivered
       after the first chunk. It isolates output delivery after startup.
   * - ``audio_before_commit_ratio``
     - Fraction of output PCM received before the final text input was sent.
   * - ``duplex_overlap_observed``
     - Whether a complete playable frame arrived before final text input.
   * - ``required_startup_delay_ms``
     - Smallest fixed playback delay that would eliminate all underruns in the
       captured stream.
   * - ``zero_delay_*``
     - Stall count and duration when playback starts immediately.
   * - ``user_audio_fluidity_index``
     - Fraction of accepted fixed-frame playback deadlines, including stalls
       caused anywhere in the end-to-end path.
   * - ``tts_service_fluidity_index``
     - The same score only when the timeline provides enough evidence to blame
       misses on TTS rather than missing upstream text.

Fluidity is inspired by the deadline, slack, and reset semantics in `Etalon
<https://arxiv.org/abs/2407.07000>`_. Veeksha first converts raw PCM supply into
fixed playback frames (20 ms by default). Early frames accumulate playable
buffer. A late frame consumes that buffer; if it is still late, every elapsed
20 ms playback deadline is a miss and the buffer resets. The score is accepted
deadlines divided by all deadlines.

The primary score defaults to zero artificial startup delay. Configure
``fluidity_startup_delay_ms`` only when the real playback client deliberately
buffers by that amount, and always report the delay with the score. Veeksha also
emits policy-specific scores such as
``user_audio_fluidity_index_d100ms`` when 100 ms is included in
``startup_delay_ms_values``.

In duplex mode, no provider-independent method can know whether a silent period
means that TTS stalled or that the upstream LLM supplied no synthesis-eligible
text. Therefore:

- ``user_audio_fluidity_index`` is always the observed user experience.
- In ``conservative`` attribution mode, service fluidity is emitted only when
  all text arrived before playback began.
- ``source_oversupplied`` may be used only by a controlled workload that
  guarantees enough eligible text throughout playback.

Batch HTTP output has all audio buffered when playback begins, so its fluidity
is trivially one. Compare batch models on first-playable latency, end-to-end
latency, RTF, cost, and quality; do not use batch fluidity to rank streaming
behavior.

Trace sources
-------------

For publishable TTS quality runs, use ``seed_tts_text``. Its current default is
the English split of the ``TwinkStart/Seed-TTS-Eval`` Hugging Face mirror. Each
row supplies target synthesis text; Veeksha records the dataset, subset, split,
and source row in request metadata. The original benchmark is maintained by
`BytedanceSpeech/seed-tts-eval
<https://github.com/BytedanceSpeech/seed-tts-eval>`_. Pin or locally archive the
exact dataset revision used in a published comparison.

``sharegpt`` is also supported when a local ShareGPT-format JSON/JSONL file is
provided. It extracts assistant turns as TTS text. Veeksha does not ship a
ShareGPT dataset.

There is no bundled Claude Code trace and no Claude-specific trace flavor.
``timed_synthetic_session`` can replay a privacy-safe coding-assistant trace of
token lengths, dependencies, and think times, but that is an LLM serving
workload—not a canonical TTS quality corpus.

Run a benchmark
---------------

This complete example uses the Seed-TTS text trace and ElevenLabs streaming.
Set ``ELEVENLABS_API_KEY`` in the environment and replace ``voice_id``.

.. code-block:: yaml

    seed: 42
    output_dir: benchmark_output/tts_elevenlabs_streaming

    client:
      type: elevenlabs_streaming_tts
      api_base: https://api.elevenlabs.io
      model: eleven_flash_v2_5
      voice_id: YOUR_VOICE_ID
      api_key_env: ELEVENLABS_API_KEY
      sample_rate: 24000
      pacing:
        tokens_per_second: 20
        tokens_per_delta: 1
        gap_distribution: fixed

    session_generator:
      type: trace
      wrap_mode: false
      flavor:
        type: seed_tts_text
        min_tokens: 20
        max_tokens: 150

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 1
      rampup_seconds: 0

    evaluators:
      - type: performance
        target_channels: [audio]
        audio_channel:
          interactivity_enabled: true
          fluidity_frame_ms: 20
          fluidity_startup_delay_ms: 0
          startup_delay_ms_values: [0, 100, 300]
          fluidity_attribution_mode: conservative
          persist_raw_timing: true
        slos:
          - type: constant
            name: P90 first playable audio under 1 second
            metric: request_start_to_first_playable_audio_ms
            percentile: 0.90
            value: 1000
          - type: constant
            name: P1 user fluidity at least 0.99
            metric: user_audio_fluidity_index
            percentile: 0.01
            value: 0.99
      - type: audio_quality
        target_channels: [audio]
        save_audio_files: true
        verification:
          wer:
            enabled: true
            threshold: 0.05
            whisper:
              model: large-v3
              device: cpu
              compute_type: int8
          utmos:
            enabled: false

    runtime:
      max_sessions: 100
      benchmark_timeout: 1800

Run it with:

.. code-block:: console

    uvx -p 3.14t veeksha benchmark --config tts_benchmark.veeksha.yml

WER requires the optional ``audio-verification`` dependencies (including
faster-whisper). UTMOS requires its corresponding optional dependency group.
Install those groups in the Veeksha environment before enabling the quality
checks.

For Deepgram Flux streaming, replace only the client block:

.. code-block:: yaml

    client:
      type: deepgram_flux_streaming_tts
      api_base: https://api.deepgram.com
      model: flux-alexis-en
      api_key_env: DEEPGRAM_API_KEY
      sample_rate: 24000
      pacing:
        tokens_per_second: 20
        tokens_per_delta: 1
        gap_distribution: fixed

Use ``deepgram_aura_streaming_tts`` with an Aura model for the ``/v1/speak``
lane. Use ``elevenlabs_http_tts`` or ``deepgram_flux_http_tts`` for the
complete-text/complete-audio controls. For Vajra, use ``realtime_tts`` and set
``input_output_mode: duplex``; the server must consume conversation items added
after an active ``response.create``.

Keep the trace seed, input texts, pacing, PCM format, region, concurrency sweep,
and retry policy identical across providers. Report request failures and rate
limits rather than silently retrying them away.

Outputs
-------

The performance evaluator writes aggregate summaries, request-level JSONL, CDF
CSVs/plots, and optional ``audio_raw_timing.jsonl``. The quality evaluator writes
WAV files and verification summaries. Provider API keys, local ``.env`` files,
downloaded traces, and benchmark output directories are not source artifacts and
must not be committed.
