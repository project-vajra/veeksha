Trace Flavors
=============

Use ``session_generator.type: trace`` when your workload should come from an
existing trace file instead of a synthetic session generator. The trace
``flavor`` tells Veeksha how to:

- validate the input file
- interpret the trace structure
- reconstruct prompts, history, and timing

.. code-block:: yaml

    session_generator:
      type: trace
      trace_file: traces/my_trace.jsonl
      wrap_mode: true
      flavor:
        type: request_log

All flavors support JSONL. CSV is also supported for simple tabular traces, and
Veeksha automatically normalizes ``num_prefill_tokens`` to ``input_length`` and
``num_decode_tokens`` to ``output_length`` when those CSV column names are used.


Choose a flavor
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 30 24 28

   * - Flavor
     - What the trace already contains
     - What Veeksha preserves
     - What Veeksha reconstructs
   * - ``request_log``
     - Independent request lengths with no real prompt text
     - Input and output length distribution only
     - Random prompt text and single-request sessions
   * - ``timed_synthetic_session``
     - Per-request token counts plus session topology and wait times
     - Linear or DAG structure, think time, and history lineage
     - Prompt text rebuilt from lengths and lineage seeds
   * - ``untimed_content_multi_turn``
     - Real user and assistant message content from a chat dataset
     - Turn order and actual conversation text
     - Request history and output lengths from assistant replies
   * - ``shared_prefix``
     - Token counts plus privacy-safe ``hash_ids`` for shared prefixes
     - Session order and cross-session prefix-sharing structure
     - Shared root prompts from ``hash_ids`` and later turns from lengths
   * - ``rag``
     - Real RAG prompt text plus repeated document IDs
     - Actual prompt text and hot-document frequency
     - Warmup sessions for the top documents

The main distinction is what survives in the trace:

- only request lengths
- lengths plus timing and topology
- actual conversation text
- privacy-safe shared-prefix identifiers
- real RAG prompts plus document reuse

If you are deciding between the multi-turn flavors:

- Use ``untimed_content_multi_turn`` when you have real chat content.
- Use ``timed_synthetic_session`` when you have timing, topology, and lengths but not prompt text.
- Use ``shared_prefix`` when you need privacy-safe traces that still preserve prefix-sharing structure.
- Use ``rag`` when you have real retrieval prompts and want cache warmup tied to repeated documents.


Common trace fields
-------------------

Several flavors reuse the same field names:

``session_id``
    Groups multiple rows into one session. Required for multi-row session
    flavors such as ``timed_synthetic_session`` and ``shared_prefix``.

``input_length``
    Total prompt tokens expected when the request is dispatched.

``new_input_length``
    New tokens introduced by this turn or node. This matters when Veeksha
    reconstructs prompts from lengths instead of real text.

``output_length``
    Requested output tokens for the request.

For flavors that expect one row per request or one row per conversation,
omitting ``session_id`` is usually safest. Veeksha will assign one
automatically.


``request_log``
----------------

Use this for the simplest replay case: independent requests where only input
and output lengths matter.

Best for:

- replaying public length-only datasets
- matching a prompt-length and output-length distribution
- load tests that do not need real multi-turn structure

Expected trace shape:

- One row per request
- Omit ``session_id`` or keep it unique per row
- Required columns: ``input_length``, ``output_length``

Minimal JSONL example with a shared root across sessions:

.. code-block:: json

    {"input_length": 512, "output_length": 128}
    {"input_length": 1024, "output_length": 256}

What Veeksha does:

- creates one single-request session per row
- generates a random prompt with ``input_length`` tokens
- requests ``output_length`` tokens from the model


``timed_synthetic_session``
---------------------------

Use this when you have recorded multi-turn or DAG-shaped workloads with timing
and token counts, but you do not want to store real prompt text.

Best for:

- coding-assistant traces
- production chat or agent traces
- KV-cache and history-lineage experiments

Expected trace shape:

- Multiple rows per session
- Required columns: ``session_id``, ``input_length``, ``new_input_length``,
  ``output_length``
- Recommended modern format: include ``session_context`` on every row

Minimal DAG-shaped JSONL example:

.. code-block:: json

    {"session_id": 8, "input_length": 8, "new_input_length": 8, "output_length": 4, "session_context": {"node_id": 0, "parent_nodes": [], "history_parent": null, "wait_after_ready": 0.0}}
    {"session_id": 8, "input_length": 8, "new_input_length": 8, "output_length": 4, "session_context": {"node_id": 1, "parent_nodes": [], "history_parent": null, "wait_after_ready": 0.1}}
    {"session_id": 8, "input_length": 16, "new_input_length": 8, "output_length": 5, "session_context": {"node_id": 2, "parent_nodes": [0, 1], "history_parent": 1, "wait_after_ready": 0.2}}

``session_context`` means:

- ``node_id``: stable request ID inside the session
- ``parent_nodes``: dependencies that must complete first
- ``history_parent``: the one parent whose history becomes this request's context
- ``wait_after_ready``: think time after dependencies complete

Legacy linear traces are also supported. If ``session_context`` is missing on
every row in a session, Veeksha falls back to row order (or ``turn_idx`` when
present) and can use ``wait_after_previous_response_s`` for linear think time.

What Veeksha does:

- rebuilds prompt text from lengths instead of using stored content
- preserves graph structure and per-node wait times
- reuses lineage seeds when two nodes share the same ``history_parent``


``untimed_content_multi_turn``
------------------------------

Use this when your trace already contains the actual conversation text and you
want Veeksha to split it into request turns.

Best for:

- ShareGPT-style datasets
- LMSYS-Chat-style datasets
- replaying real prompt text without timestamps

Expected trace shape:

- One row per conversation
- Omit ``session_id`` or keep it unique per row
- Required column: the conversation column, ``conversations`` by default

Minimal ShareGPT-style JSONL example:

.. code-block:: json

    {"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "Python is a programming language."}, {"from": "human", "value": "Tell me more."}, {"from": "gpt", "value": "It was first released in 1991."}]}

Minimal LMSYS-style flavor config:

.. code-block:: yaml

    flavor:
      type: untimed_content_multi_turn
      conversation_column: conversation
      role_key: role
      content_key: content
      user_role_value: user
      assistant_role_value: assistant

What Veeksha does:

- turns each user/assistant pair into one request
- pre-populates request history from earlier messages in the row
- skips leading assistant messages, system messages, and trailing user messages without a response


``shared_prefix``
-----------------

Use this when you need privacy-safe traces that preserve shared-prefix behavior
across sessions.

Best for:

- testing prefix caching without storing real text
- replaying workloads where many sessions share common roots
- synthetic reconstruction from stable prefix IDs

Expected trace shape:

- Multiple rows per session
- Rows within a session should already be in turn order
- Required columns: ``session_id``, ``input_length``, ``new_input_length``,
  ``output_length``, ``hash_ids``

Minimal JSONL example:

.. code-block:: json

    {"session_id": 1, "input_length": 1024, "new_input_length": 1024, "output_length": 128, "hash_ids": [101, 102]}
    {"session_id": 1, "input_length": 1536, "new_input_length": 512, "output_length": 128, "hash_ids": [101, 102, 103]}
    {"session_id": 2, "input_length": 1024, "new_input_length": 1024, "output_length": 128, "hash_ids": [101, 102]}

What Veeksha does:

- uses the first row of each session as the shared-prefix root request
- maps identical root ``hash_ids`` blocks to identical prompt content
- uses ``hash_ids`` for cross-session sharing only on that first row
- generates later-turn content from ``new_input_length``

In practice, this flavor is useful when you care about shared prefixes and
conversation structure, but cannot keep the original prompt text in the trace.


``rag``
-------

Use this for retrieval-style workloads where the trace already contains the full
prompt text and repeated document IDs.

Best for:

- RAG traces with repeated retrieved documents
- document-cache warmup experiments
- long-context single-request workloads

Expected trace shape:

- One row per request
- Omit ``session_id`` or keep it unique per row
- Required columns: ``doc_id``, ``prompt_text``, ``input_length``,
  ``output_length``

Minimal JSONL example with a repeated hot document:

.. code-block:: json

    {"doc_id": "doc-17", "prompt_text": "Context: ...\n\nQuestion: Summarize the document.", "input_length": 2048, "output_length": 128}
    {"doc_id": "doc-17", "prompt_text": "Context: ...\n\nQuestion: Extract the main risks.", "input_length": 2048, "output_length": 128}
    {"doc_id": "doc-42", "prompt_text": "Context: ...\n\nQuestion: List the action items.", "input_length": 2048, "output_length": 128}

What Veeksha does:

- creates one single-request session per row
- picks the most frequent ``doc_id`` values for warmup sessions
- warms those documents before the main benchmark starts

Tune ``flavor.num_documents`` to control how many hot documents Veeksha warms.


Wrap mode
---------

With ``wrap_mode: true``, Veeksha keeps looping over the trace after it reaches
the end. Sessions are reshuffled between epochs, and flavors that synthesize
content regenerate or remap it as needed so repeated epochs do not reuse exactly
the same session IDs and prompt materialization.


See also
--------

- :doc:`configuration` for full trace-session-generator configuration
- :doc:`/getting_started/common_benchmarks` for ready-to-run benchmark recipes
- :doc:`/design/content_generation` for the design-level view of content generation
