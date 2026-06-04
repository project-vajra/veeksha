Traffic Scheduling
==================

Traffic scheduling controls **when** sessions start and **when** requests within
sessions are dispatched. Veeksha provides two fundamentally different scheduling
modes for different benchmarking scenarios.


Scheduling modes
----------------

**Rate-Based** (``type: rate``)
    Generates new sessions at a specified arrival rate, regardless of how many
    are currently in-flight. Models open-loop traffic.

**Concurrency-Based** (``type: concurrent``)
    Maintains a target number of active sessions. When one completes, another
    starts. Models closed-loop traffic.

.. list-table:: When to Use Each Mode
   :header-rows: 1
   :widths: 30 35 35

   * - Scenario
     - Mode
     - Rationale
   * - Latency under load
     - Rate-based
     - Measure how latency degrades as rate increases
   * - Maximum throughput
     - Concurrent
     - Saturate the system to find peak capacity
   * - Production traffic modeling
     - Rate-based (Poisson)
     - Poisson arrivals model realistic bursty traffic
   * - Capacity planning
     - Rate-based
     - Find the rate where latency SLOs are met
   * - Stress testing
     - Concurrent (high)
     - Push beyond normal operating conditions


Rate-based scheduling
---------------------

Sessions arrive according to an interval generator:

.. code-block:: yaml

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0
      cancel_session_on_failure: true

**How it works:**

1. `RateTrafficScheduler` generates inter-arrival times from the interval generator
2. Each session's root requests are scheduled at the computed arrival time
3. Sessions are dispatched regardless of current system load

**Interval Generators:**

``poisson`` (recommended for realism)
    Exponentially-distributed intervals with given mean rate:

    .. code-block:: yaml

        interval_generator:
          type: poisson
          arrival_rate: 10.0  # 10 sessions/second average

    Captures real-world bursty arrival patterns.

``gamma``
    Gamma-distributed intervals (generalization of Poisson):

    .. code-block:: yaml

        interval_generator:
          type: gamma
          arrival_rate: 10.0
          shape: 2.0  # Higher = less variance

``fixed``
    Constant intervals for uniform traffic:

    .. code-block:: yaml

        interval_generator:
          type: fixed
          interval: 0.1  # Exactly 100ms between sessions


Concurrency-based scheduling
----------------------------

Maintains a fixed number of concurrent sessions:

.. code-block:: yaml

    traffic_scheduler:
      type: concurrent
      target_concurrent_sessions: 8
      rampup_seconds: 10
      cancel_session_on_failure: true

**How it works:**

1. `ConcurrentTrafficScheduler` tracks active session count
2. When a session completes, it activates a pending one
3. Ramp-up gradually increases concurrency from 0 to target

**Ramp-up Behavior:**

.. code-block:: text

    Concurrency
        ▲
        │                    ┌────────────────────
      8 │                   ╱
        │                  ╱
      4 │                 ╱
        │                ╱
      0 │───────────────╱
        └──────────────────────────────────────▶ Time
        0            10s (rampup)         ...

During ramp-up, target concurrency increases linearly:

.. code-block:: python

    current_target = int(target * (elapsed_time / rampup_seconds))


Intra-session scheduling
------------------------

Within a session, requests are scheduled based on the session graph:

.. code-block:: text

    Session with 3 turns:
    
    t=0.0s: Root request dispatched (session arrives)
    t=1.2s: Root request completes
    t=1.7s: Turn 1 dispatched (0.5s wait_after_ready)
    t=2.1s: Turn 1 completes
    t=2.4s: Turn 2 dispatched (0.3s wait_after_ready)
    ...

The scheduler tracks session state:

.. code-block:: python

    class ScheduledSessionState:
        session: Session
        completed_nodes: Set[int]      # Finished request nodes
        in_flight_nodes: Set[int]      # Currently executing
        pending_nodes: Set[int]        # Waiting on dependencies
        completion_times: Dict[int, float]  # When each node finished

When a request completes:

1. Node is moved from ``in_flight_nodes`` to ``completed_nodes``
2. Child nodes are checked for readiness
3. Ready nodes are scheduled after their ``wait_after_ready`` delay
4. History is recorded if this node is a history parent


Session cancellation
--------------------

The ``cancel_session_on_failure`` option controls behavior when a request fails:

.. code-block:: yaml

    traffic_scheduler:
      cancel_session_on_failure: true  # Default

When ``true``, if any request in a session fails:

- All pending requests in that session are cancelled
- The session is marked as errored
- Resources are freed for new sessions

When ``false``:

- Remaining requests in the session are still attempted
- Useful for testing partial failure scenarios


Ready queue and dispatch
------------------------

Both schedulers maintain a **ready queue** of requests eligible for dispatch:

.. code-block:: text

    Ready Queue (min-heap by ready_at time):
    ┌─────────────────────────────────────────┐
    │ (ready_at=0.0, request_1)              │ ← Pop next
    │ (ready_at=0.1, request_5)              │
    │ (ready_at=0.2, request_3)              │
    │ (ready_at=0.5, request_8)              │
    └─────────────────────────────────────────┘

Dispatch workers call ``wait_for_ready()`` which:

1. Waits until the next ready time (or timeout)
2. Pops the request and marks it dispatched
3. Records ``scheduler_dispatched_at`` timestamp

This ensures requests are dispatched at the right time (not early, not late).


History population
------------------

When ``inherit_history: true`` in the session graph, the scheduler populates
request history from parent responses:

.. code-block:: python

    def _populate_history(self, request: Request, state: ScheduledSessionState, node_id: int):
        """Populate request history from parent nodes."""
        for edge in parents(state.session.session_graph, node_id):
            if edge.is_history_parent:
                parent_history = state.histories.get(edge.src)
                if parent_history:
                    request.populate_history(parent_history)

The history includes:

- Prior request content (prompts)
- Prior response content (model outputs)
- Enables accurate multi-turn conversation simulation


Timing verification
-------------------

Veeksha's health checker verifies scheduling accuracy:

**Session Dispatch Rate Check**
    Compares actual vs configured arrival rate:

    .. code-block:: text

        Expected Rate: 10.0 sessions/sec
        Actual Rate: 10.2 sessions/sec
        Error: 2.0%
        Threshold: 15%
        Result: PASSED

**Intra-Session Request Arrival Check**
    Verifies requests weren't dispatched before dependencies completed:

    .. code-block:: text

        Requests w/ Dependencies: 445
        Mean Delay: 0.0017s
        P99 Delay: 0.0788s
        Violations (>5s late): 0
        Result: PASSED

These checks help identify issues with benchmark configuration or execution.
