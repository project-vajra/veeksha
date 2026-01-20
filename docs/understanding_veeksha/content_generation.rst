Content Generation
==================

Veeksha uses a channel-based content generation system that supports multiple
modalities (text, images, audio, video) and provides fine-grained control over
request characteristics.


Channel architecture
--------------------

Request content is organized by **channel modality**:

.. code-block:: python

    class ChannelModality(IntEnum):
        TEXT = 1
        IMAGE = 2
        AUDIO = 3
        VIDEO = 4

Each request contains content for one or more channels:

.. code-block:: python

    @dataclass
    class Request:
        channels: Dict[ChannelModality, ChannelContent]
        # e.g., {ChannelModality.TEXT: TextContent(...)}

Currently, the **text channel** is fully implemented with others planned.


Text channel generator
----------------------

The text channel generator produces prompt content with configurable lengths
and optional shared prefixes:

.. code-block:: yaml

    session_generator:
      type: synthetic
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500
          shared_prefix_ratio: 0.2
          shared_prefix_probability: 0.5
      output_spec:
        text:
          output_length_generator:
            type: uniform
            min: 50
            max: 200

Key configuration options:

``body_length_generator``
    Controls the number of tokens in the prompt body (new content per turn).

``shared_prefix_ratio``
    Fraction of prompt tokens that should be identical across root requests.
    Useful for testing prefix caching.

``shared_prefix_probability``
    Probability that a root request uses the shared prefix.

Output specification is configured separately at the session generator level. Specified output
specs will only be relevant if the model supports the modality. For example:

``output_spec.text.output_length_generator``
    Controls the requested output length (``max_tokens`` / ``min_tokens``).


Length generators
-----------------

Length generators control numeric parameters like token counts:

**Fixed** (``type: fixed``)
    Returns a constant value:

    .. code-block:: yaml

        body_length_generator:
          type: fixed
          value: 256

**Uniform** (``type: uniform``)
    Random value in a range:

    .. code-block:: yaml

        body_length_generator:
          type: uniform
          min: 100
          max: 500

**Stair** (``type: fixed_stair``)
    Cycles through explicit values in order, useful for microbenchmarking:

    .. code-block:: yaml

        body_length_generator:
          type: fixed_stair
          values: [128, 256, 512, 1024]  # Values to cycle through
          repeat_each: 10                 # Repetitions per value
          wrap: true                      # Cycle back to start

    This generates: 10 requests at 128, then 10 at 256, then 10 at 512, etc.

**Zipf** (``type: zipf``)
    Power-law distribution modeling real-world length patterns:

    .. code-block:: yaml

        body_length_generator:
          type: zipf
          min: 50
          max: 2000
          alpha: 1.5


Content generation process
--------------------------

When a synthetic session is generated:

1. **Session graph** is created with the configured number of nodes
2. For each node, **channels** generate content:

   .. code-block:: python

       for channel_type, channel in self.channels.items():
           channels[channel_type] = channel.generate_content(
               is_root=is_root(session_graph, node_id)
           )

3. The ``is_root`` flag enables special handling (e.g., shared prefixes apply
   only to root requests)

4. The **output specification** is generated via ``OutputSpecGenerator``
   and attached to each request. This includes target output tokens.


Shared prefix for prefix caching
--------------------------------

Most LLM inference engines support **prefix caching** where repeated
prompt prefixes are cached in KV cache. Veeksha can generate workloads that
test this. First, by setting a shared prefix configuration:

.. code-block:: yaml

    channels:
      - type: text
        shared_prefix_ratio: 0.3
        shared_prefix_probability: 0.8

This configuration means:

- 80% of root requests will share a common prefix
- That prefix constitutes 30% of the total prompt tokens

When generating content:

1. A single shared prefix is generated once and cached
2. Root requests probabilistically use this prefix
3. The remaining tokens are generated uniquely per request

This accurately models scenarios like:

- System prompts shared across users
- RAG with common document prefixes
- Function calling with shared tool definitions

Another way in which Veeksha helps test prefix cache capabilities is by making session nodes inherit conversation
history. This is done by setting the ``inherit_history`` flag to ``true`` in the session generator configuration:

.. code-block:: yaml

    session_generator:
      type: synthetic
      inherit_history: true

A node can only inherit history from one of its parent nodes. 

Tokenizer integration
---------------------

Content generation requires tokenization to control token counts precisely.
Veeksha uses a **TokenizerProvider** pattern:

.. code-block:: python

    class TokenizerProvider:
        """Provides tokenizers for different modalities."""

        def for_modality(self, modality: ChannelModality) -> TokenizerHandle:
            ...

For text, this wraps a HuggingFace tokenizer (loaded based on the model name).
The tokenizer is used to:

1. Encode generated text to count tokens
2. Decode token IDs for prompt construction
3. Ensure prompt lengths match targets exactly

.. note::

    When running a benchmark, ensure the tokenizer matches the model being
    tested. Veeksha loads the tokenizer automatically based on
    ``client.model`` or ``server.model``.


Trace-based content
-------------------

For trace-based session generation, content comes from recorded conversations
stored in JSONL files. Each trace file contains pre-recorded prompts and
metadata matching real production traffic.

.. code-block:: yaml

    session_generator:
      type: trace
      trace_file: traces/claude_code.jsonl
      flavor:
        type: claude_code


Trace flavors
~~~~~~~~~~~~~

Different trace sources have different formats and characteristics. **Flavors**
define how to parse trace files and generate sessions from them. Each flavor
implements:

- Required column validation
- Session/request preparation from trace rows
- Wrapping behavior for looping through traces

**claude_code** (``type: claude_code``)
    Context-cached coding assistant traces with these characteristics:

    - Multi-turn conversations with history accumulation
    - Unique prompts generated via hashing for KV-cache diversity
    - Wait times between turns preserved from trace
    - Session prefix seeds for reproducible prompt generation

    Required columns: ``session_id``, ``input_length``, ``output_length``

    .. code-block:: yaml

        flavor:
          type: claude_code
          page_size: 16          # Token page size for prefix caching
          corpus_file: null      # Optional corpus for prompt generation

**rag** (``type: rag``)
    Retrieval-Augmented Generation workload traces:

    - Single-turn requests (one request per session)
    - Document-based filtering by frequency
    - Warmup sessions to pre-populate document cache
    - Suitable for testing prefix caching with shared documents

    Required columns: ``doc_id``, ``prompt_text``, ``input_length``, ``output_length``

    .. code-block:: yaml

        flavor:
          type: rag
          num_documents: 10      # Use top N most frequent documents

**mooncake_conv** (``type: mooncake_conv``)
    Mooncake conversational dataset traces:

    - Multi-turn conversations with varying session lengths
    - Direct prompt text from trace (no generation)

    Required columns: ``session_id``, ``prompt_text``, ``output_length``


Wrap mode
~~~~~~~~~

When ``wrap_mode: true`` (default), the trace loops indefinitely:

.. code-block:: yaml

    session_generator:
      type: trace
      trace_file: traces/production.jsonl
      wrap_mode: true    # Loop through trace when exhausted

On wrap, traces are reshuffled to provide different orderings. This enables
running benchmarks longer than the trace duration while maintaining realistic
content distributions.


Output length control
---------------------

Veeksha supports flexible output length control:

**Server-side** (preferred when supported):

.. code-block:: yaml

    client:
      max_tokens_param: max_completion_tokens
      min_tokens_param: min_tokens

The generator sets both ``max_tokens`` and ``min_tokens`` to the target,
forcing exact output lengths when the server supports it.

**Prompt-based fallback**:

.. code-block:: yaml

    client:
      use_min_tokens_prompt_fallback: true

Appends instructions like "Generate exactly 150 tokens" to the prompt.
Less reliable but works with servers lacking ``min_tokens`` support.

.. tip::

    For accurate benchmarks, use a server that supports ``min_tokens`` to control output lengths precisely.
