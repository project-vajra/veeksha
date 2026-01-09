Sessions and Graphs
===================

Veeksha models LLM interactions as **sessions** containing **requests** organized
in a directed acyclic graph (DAG). This design captures the dependency structure
of multi-turn conversations.


The Session Model
-----------------

A **Session** represents a complete user conversation and contains:

- A unique session ID
- A **SessionGraph** defining the structure of requests
- A dictionary of **Request** objects keyed by node ID

.. code-block:: python

    @dataclass
    class Session:
        id: int
        session_graph: SessionGraph
        requests: Dict[int, Request]  # node_id -> Request

A **Request** represents a single interaction (prompt + expected response):

.. code-block:: python

    @dataclass
    class Request:
        id: int                                    # Unique global request ID
        channels: Dict[ChannelModality, Content]   # Content per modality
        session_context: Dict[str, Any]            # Graph metadata


Session Graphs as DAGs
----------------------

The **SessionGraph** models request dependencies using nodes and directed edges:

.. code-block:: text

    Linear Session (3 turns):
    
    ┌───────┐      ┌───────┐      ┌───────┐
    │ Node 0│─────▶│ Node 1│─────▶│ Node 2│
    └───────┘      └───────┘      └───────┘
    (root)         wait: 0.5s     wait: 0.3s
    
    Edges indicate dependencies:
    - Node 1 cannot start until Node 0 completes
    - Node 2 cannot start until Node 1 completes

Each **SessionNode** contains:

- ``id``: Node identifier within the session
- ``wait_after_ready``: Delay (in seconds) after dependencies are satisfied

Each **SessionEdge** contains:

- ``src``, ``dst``: Source and destination node IDs
- ``is_history_parent``: Whether parent's output should be included in context


Linear Sessions
---------------

The most common pattern is a **linear session** representing a typical
back-and-forth conversation:

.. code-block:: yaml

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        num_request_generator:
          type: uniform
          min: 2
          max: 6
        request_wait_generator:
          type: poisson
          arrival_rate: 1.0
        inherit_history: true

Configuration options:

``num_request_generator``
    Controls how many turns (requests) each session contains.
    Supports distributions: ``fixed``, ``uniform``, ``zipf``, ``stair``.

``request_wait_generator``
    Controls the "think time" between turns-how long after one request
    completes before the next is dispatched.
    Supports: ``fixed``, ``poisson``, ``gamma``.

``inherit_history``
    If ``true``, each request includes the conversation history from
    its parent node(s), simulating chat context accumulation.


History Inheritance
-------------------

When ``inherit_history: true``, the traffic scheduler populates each request's
history based on edges marked as ``is_history_parent``:

.. code-block:: text

    Turn 0: "What is Python?"
        ↓ (history edge)
    Turn 1: "What is Python?" → "Python is..." + "Tell me more"
        ↓ (history edge)
    Turn 2: [full history] + "Give me an example"

The history is recorded when a request completes and includes:

- The request content (prompt)
- The response content (model output)
- Timing information

This accurately models how LLM chat APIs accumulate conversation context.


Branching Graphs (Future)
-------------------------

While linear sessions cover most use cases, the DAG structure supports
more complex patterns:

.. code-block:: text

    Branching Session:
    
         ┌───────┐
         │ Root  │
         └───┬───┘
             │
        ┌────┴────┐
        ▼         ▼
    ┌───────┐ ┌───────┐
    │ Path A│ │ Path B│
    └───────┘ └───────┘

This could model scenarios like:

- Parallel tool calls
- A/B testing different conversation paths
- Multi-agent interactions

.. note::

    Currently, only the ``linear`` graph generator is implemented.
    The architecture supports future graph types.


Session Generators
------------------

Three session generator types are available:

**Synthetic** (``type: synthetic``)
    Generates sessions with random but controlled content. Combines:

    - A session graph generator (linear)
    - Channel generators for request content

    Best for: Load testing with configurable workload characteristics.

**Trace** (``type: trace``)
    Replays recorded conversation traces from JSONL files:

    .. code-block:: yaml

        session_generator:
          type: trace
          trace_file: traces/claude_code.jsonl
          flavor:
            type: claude_code
          wrap_mode: true

    Supported trace flavors:

    - ``claude_code``: Claude Code assistant traces
    - ``mooncake_conv``: Mooncake conversation dataset
    - ``rag``: RAG (Retrieval-Augmented Generation) traces

    Best for: Realistic workload replay, production traffic analysis.

**LM-Eval** (``type: lmeval``)
    Generates evaluation prompts from lm-eval-harness tasks:

    .. code-block:: yaml

        session_generator:
          type: lmeval
          tasks: ["hellaswag", "truthfulqa_gen"]
          num_fewshot: 5

    Best for: Model accuracy evaluation under load.


Request Scheduling Within Sessions
----------------------------------

When a session is scheduled, its requests don't all dispatch immediately.
The traffic scheduler respects the graph structure:

1. **Root nodes** (no incoming edges) are immediately ready
2. **Dependent nodes** wait for all parent nodes to complete
3. After parents complete, ``wait_after_ready`` delay is observed
4. Only then is the request marked ready for dispatch

This is handled by the ``ScheduledSessionState`` class which tracks:

- Completed node IDs
- Pending node IDs
- Per-node completion times and history

The health checker verifies this timing with the
"Intra-Session Request Arrival Check" that validates requests weren't
dispatched before their dependencies completed.
