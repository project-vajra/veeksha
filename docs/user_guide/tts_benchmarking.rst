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

The public client type describes the transport. The ``provider`` field selects
the wire protocol behind that transport, while every provider shares the same
request lifecycle and metric contract:

.. list-table::
   :header-rows: 1

   * - ``client.type``
     - ``provider``
     - Transport
     - Input behavior
   * - ``tts``
     - ``openai``
     - OpenAI-compatible ``POST /v1/audio/speech``
     - Complete text with an HTTP audio response
   * - ``tts``
     - ``elevenlabs``
     - ElevenLabs ``/v1/text-to-speech`` HTTP
     - Complete text with an HTTP PCM response
   * - ``tts``
     - ``deepgram_flux``
     - Deepgram Flux ``POST /v2/speak``
     - Complete text with an HTTP PCM response
   * - ``streaming_tts``
     - ``openai_realtime``
     - OpenAI Realtime-compatible WebSocket
     - Complete-text or duplex ``response.create`` scheduling
   * - ``streaming_tts``
     - ``vajra``
     - Vajra native ``/v1/audio/speech/stream`` WebSocket
     - Paced ``input.text`` messages with binary PCM output
   * - ``streaming_tts``
     - ``elevenlabs``
     - ElevenLabs ``/stream-input`` WebSocket
     - Paced partial text followed by provider finalization
   * - ``streaming_tts``
     - ``deepgram_flux``
     - Deepgram Flux ``/v2/speak`` WebSocket
     - Paced ``Speak`` messages; audio may arrive before ``Flush``
   * - ``streaming_tts``
     - ``deepgram_aura``
     - Deepgram Aura ``/v1/speak`` WebSocket
     - Paced ``Speak`` messages followed by ``Flush``

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
   * - ``trigger_to_first_playable_audio_ms``
     - Synthesis trigger to the first complete PCM playback frame on the active
       connection. This is the primary steady-state TTFA: ``response.create``
       for Realtime TTS and the first real synthesis-eligible text message for
       native streaming APIs. Protocol setup messages are excluded.
   * - ``first_input_to_first_playable_audio_ms``
     - First real streamed text delta to the first complete PCM playback frame.
       Unlike trigger TTFA, this intentionally exposes any client or provider
       lookahead before synthesis is triggered.
   * - ``request_start_to_first_playable_audio_ms``
     - WebSocket-connect initiation to the first complete PCM playback frame.
       Report this separately as cold-session or connection-inclusive latency.
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

.. literalinclude:: ../../veeksha/sample_configs/tts_streaming_elevenlabs.yml
   :language: yaml
   :caption: veeksha/sample_configs/tts_streaming_elevenlabs.yml

Run it with:

.. code-block:: console

    uvx -p 3.14t veeksha benchmark \
      --config veeksha/sample_configs/tts_streaming_elevenlabs.yml

WER requires the optional ``audio-verification`` dependencies (including
faster-whisper). UTMOS requires its corresponding optional dependency group.
Install those groups in the Veeksha environment before enabling the quality
checks.

Correctness metric contract
---------------------------

ASR and TTS WER intentionally use different published protocols and therefore
different units:

- ASR ``final_wer`` and aggregate ``asr_*_wer`` fields are percentages in the
  range 0--100 for ordinary cases. Text is normalized with the Open ASR
  Leaderboard English normalizer. Prefer corpus WER for the primary comparison;
  sample-mean and duration-weighted WER are reported as diagnostics.
- TTS verification ``wer`` fields are ratios where 0.05 means five percent.
  They use Seed-TTS-style punctuation normalization and a configured
  faster-whisper judge. Keep the judge checkpoint, language, beam size, device,
  compute type, corpus revision, and text normalization identical across every
  provider. This WER measures intelligibility, not naturalness, speaker
  similarity, emotion, or human preference.
- UTMOS is a predicted naturalness score. Treat it as a scalable regression
  signal, not a substitute for a blinded human preference study.

With ``fail_on_threshold: true``, verification fails closed: a WER threshold
violation, missing audio file, transcription failure, unavailable UTMOS model,
or run-level verification error fails the benchmark rather than silently
removing that request from the quality sample.

For Deepgram Flux streaming, replace only the client block:

.. code-block:: yaml

    client:
      type: streaming_tts
      provider: deepgram_flux
      api_base: https://api.deepgram.com
      model: flux-alexis-en
      api_key_env: DEEPGRAM_API_KEY
      sample_rate: 24000
      pacing:
        tokens_per_second: 20
        tokens_per_delta: 1
        gap_distribution: fixed

For Aura, keep ``type: streaming_tts`` and set ``provider: deepgram_aura``.
For complete-text HTTP controls, use ``type: tts`` with either
``provider: elevenlabs`` or ``provider: deepgram_flux``. The OpenAI-compatible
HTTP speech contract is ``type: tts`` with ``provider: openai``.

Vajra's native streaming contract uses ``type: streaming_tts`` with
``provider: vajra``. A Vajra endpoint implementing the OpenAI Realtime
contract instead uses ``provider: openai_realtime``. Set
``input_output_mode: duplex`` only for the explicit Realtime response trigger; the server must
consume conversation items added after an active ``response.create``.

Do not treat ``response.create`` ordering or output overlap alone as proof of
semantic duplex synthesis. A conforming run must also pass full-reference TTS
WER (or a stronger text-coverage check) so a server that speaks only the prefix
available at trigger time cannot be reported as successful streaming. The
``duplex_start_after_tokens`` value is a configurable workload threshold, not a
special eight-token protocol rule.

Keep the trace seed, input texts, pacing, PCM format, region, concurrency sweep,
and retry policy identical across providers. Report request failures and rate
limits rather than silently retrying them away.

Adversarial abort testing
-------------------------

Vajra's native streaming provider can deliberately disconnect a deterministic
fraction of sessions partway through synthesis. This exercises server-side
abort, slot-reclamation, and staging teardown under load:

.. code-block:: yaml

    client:
      type: streaming_tts
      provider: vajra
      api_base: http://localhost:8003
      model: your-vajra-tts-model
      abort:
        fraction: 0.1
        trigger: audio_ms
        value: 1000
        seed: 1234

``trigger`` accepts ``audio_ms``, ``input_fraction``, or ``wall_clock_s``;
``value`` uses milliseconds, a fraction in ``(0, 1]``, or seconds,
respectively. Selection is deterministic for a given ``seed`` and session ID.
Abort injection is rejected for non-Vajra providers.

Intentionally aborted requests remain visible in request-level output and in
``aborted_requests_count``. They are excluded from normal audio latency,
duration, RTF, fluidity, and audio-throughput aggregates because their output is
partial by construction.

Health boundaries and live validation
-------------------------------------

If the server has a known output-length cap, set
``evaluators[].audio_channel.max_expected_audio_ms``. A non-aborted TTS request
whose generated audio reaches that duration, within one 320 ms codec chunk, is
reported as suspected length-cap truncation. For example:

.. code-block:: yaml

    evaluators:
      - type: performance
        target_channels: [audio]
        audio_channel:
          max_expected_audio_ms: 163840

Vajra zombie-session accounting is enabled only when the benchmark has a Vajra
``endpoint`` with ``health_url`` configured. The server must expose
``/debug/tts_worker_stats`` and run with ``VAJRA_TTS_TELEMETRY_DIR`` set.
Veeksha snapshots cumulative Talker completion counters before and after the
run and compares their delta with benchmark completions. A positive surplus
means disconnected clients left server-side work running during the measured
window. Missing or disabled telemetry is reported as a skipped health check,
not silently treated as measured evidence.

Offline tests cover configuration, protocol state machines, abort behavior, and
metric accounting. They cannot validate live provider availability,
credential scope, regional routing, rate limits, billed cost, or vendor-side
model changes. Run credentialed smoke tests before publishing cross-provider
results, keep provider secrets in environment variables, and record region,
model identifier, pricing date, and retry policy with the result.

Outputs
-------

The performance evaluator writes aggregate summaries, request-level JSONL, CDF
CSVs/plots, and optional ``audio_raw_timing.jsonl``. The quality evaluator writes
WAV files and verification summaries. Provider API keys, local ``.env`` files,
downloaded traces, and benchmark output directories are not source artifacts and
must not be committed.
