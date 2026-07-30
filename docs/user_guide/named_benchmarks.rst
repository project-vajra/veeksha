Named Benchmarks
================

A named benchmark is a workload you can freeze, share, and re-run with a
guarantee that the requests reaching the server are the ones that were pinned.

Comparable numbers require the workload itself to be identical across runs.
Without that guarantee a config edit, a tokenizer upgrade, or dataset drift can
quietly change what gets sent, and two runs differ for reasons that have
nothing to do with the system under test.

A definition freezes its workload behind a **blake2b fingerprint** computed
over the generated requests. Pinning records that fingerprint; running
recomputes it and refuses to proceed on a mismatch — before a single request
leaves the client.


How it works
------------

**Define (pin)**
    Generate the workload, hash it, and record the fingerprint alongside
    provenance: veeksha version and commit, tokenizer and package versions,
    and digests of any referenced assets.

**Run (verify)**
    Regenerate the workload, recompute the fingerprint, and compare it to the
    pin. A mismatch aborts the run and reports which recorded input changed.

**Free variables**
    Anything that must vary between runs is declared explicitly. ``define``
    sweeps each declared knob across its choices and fails if any of them moves
    the fingerprint, so load parameters cannot be confused with changes to the
    workload.

.. code-block:: text

    define                          run
    ──────                          ───
    generate workload               regenerate workload
    hash    → blake2b:a799…         hash    → blake2b:a799…
    verify each knob is inert       compare to pin
    write pins into benchmark.yml     match → proceed
                                      differ → abort at preflight


Anatomy of a definition
-----------------------

A definition is a directory containing ``benchmark.yml``:

.. code-block:: text

    benchmarks/synthetic-concurrency/
      benchmark.yml    # config + knobs + pins (source of truth)
      pins.json        # human-readable copy of the pins
      README.md

.. code-block:: yaml

    name: synthetic-concurrency
    version: 1
    purpose: Minimal named benchmark for the free-variable / pin workflow.

    # Free variables: the only things a run may change.
    knobs:
      concurrency:
        target: traffic_scheduler.target_concurrent_sessions
        type: int              # int | float | str | bool
        default: 1
        choices: [1, 2, 4, 8]
        help: Steady-state concurrent sessions (load only).

    # The frozen workload. Everything here is pinned.
    config:
      seed: 42
      traffic_scheduler:
        type: concurrent
        target_concurrent_sessions: 1
      session_generator:
        type: synthetic
        session_graph:
          type: single_request
        channels:
          - type: text
            body_length_generator:
              type: fixed
              value: 16
      runtime:
        max_sessions: 8
        pregenerate_sessions: true

    # Written by `benchmark define` -- do not hand-edit.
    pins:
      workload_fingerprint: blake2b:a79911b4…
      tokenizer: {model: gpt2, transformers: 5.14.1, tokenizers: 0.22.2}
      veeksha: {version: 0.4.5, git_commit: …, git_dirty: false}

.. important::

   The ``config:`` block in ``benchmark.yml`` is the source of truth. ``define``
   resolves any ``!include`` and writes the config back **inlined**, so a
   separate base file stops being read after the first pin. Edit the workload
   in this block, then re-pin.


Pinning a definition
--------------------

.. code-block:: bash

    veeksha benchmark define --definition benchmarks/synthetic-concurrency

The number of sessions hashed comes from ``config.runtime.max_sessions``. No
server is contacted — this is generation only.

Useful options:

``--definition``, ``--def``
    Path to the definition directory or its ``benchmark.yml``.

``--output``
    Write the pinned tree elsewhere instead of updating in place.

``--publish``, ``--repo``, ``--tag``, ``--private``
    Upload the pinned definition to the Hugging Face Hub after pinning.

.. note::

   ``benchmark define`` is the one command permitted on a normal (GIL-enabled)
   interpreter, because free-threaded wheels are not always available for the
   tokenizer stack. Every other command requires free-threaded Python.


Running a named benchmark
-------------------------

.. code-block:: bash

    veeksha benchmark run \
        --benchmark benchmarks/synthetic-concurrency \
        --concurrency 4 \
        --endpoint.engine_type vllm \
        --endpoint.model gpt2 \
        --endpoint.api_base http://localhost:8000/v1 \
        --output_dir runs/synthetic-concurrency

``--benchmark`` accepts a local directory, a path to ``benchmark.yml``, or a
Hub name. A Hub name is downloaded into ``~/.cache/veeksha/benchmarks``
(override with ``VEEKSHA_BENCHMARK_CACHE``); pin a revision with
``--benchmark_revision``.

Each declared knob becomes a CLI flag — ``--concurrency 4`` above.

With ``runtime.pregenerate_sessions: true`` the whole workload is hashed before
the run starts, so the pin check happens **before any request is dispatched**.


What you may override
---------------------

A named benchmark is frozen. These are allowed because they place the run
without changing the workload:

.. code-block:: text

    output_dir              endpoint.*
    benchmark               server.*
    benchmark_revision      wandb.*
    allow_config_override
    allow_workload_drift

    …plus every declared free variable

Anything else is rejected:

.. code-block:: text

    Named benchmarks are frozen: these CLI flags override definition fields
    and are not declared free variables: seed. Declared free variables:
    concurrency. Remove the flags, declare them as knobs in the definition,
    or pass --allow_config_override true (marks the run unpinned).

Two escape hatches exist, and both mark the run as no longer pinned:

``--allow_config_override true``
    Permit arbitrary CLI overrides of frozen fields.

``--allow_workload_drift true``
    Downgrade a fingerprint mismatch from an error to a warning.


When the pin does not match
---------------------------

.. code-block:: text

    Workload fingerprint mismatch for named benchmark 'synthetic-concurrency'
    at preflight: expected blake2b:a79911b4…, got blake2b:35b36542…
    Likely causes: no input diffs recorded

The run exits non-zero and no requests are sent. Either restore the definition,
or re-run ``benchmark define`` if the change was intended — re-pinning is how
you deliberately publish a new workload.

The check runs at two stages:

``preflight``
    Only when sessions are pre-generated. The whole workload is already hashed,
    so the run aborts before dispatching anything. There is no run record to
    diff against yet, hence ``no input diffs recorded``.

``finalize``
    After the run, comparing against the recorded inputs in the run manifest.
    Here the message names what actually moved, for example
    ``tokenizer model: 'gpt2' -> 'gpt2-medium'`` or
    ``transformers version: '5.14.1' -> '5.15.0'``.


Sharing definitions
-------------------

Definitions can live on the Hugging Face Hub as a dataset repo. Each benchmark
is fetched independently, so one benchmark never pulls the whole repository:

.. code-block:: bash

    # publish
    veeksha benchmark define --definition benchmarks/my-benchmark \
        --publish true --repo my-org/veeksha-benchmarks --tag v1

    # consume
    veeksha benchmark run --benchmark my-benchmark --benchmark_revision v1 …

The default repo comes from ``VEEKSHA_BENCHMARKS_REPO``.


Gotchas
-------

**Unknown config keys are dropped silently.**
    A misspelled field does not raise — it falls back to its default, and
    ``define`` then pins that workload faithfully. If a pinned value looks
    wrong, check the field name against the config reference. For example, the
    fixed length generator takes ``value``, not ``length``.

**Keep the GIL disabled.**
    Some C extensions (``tokenizers`` among them) do not declare free-threading
    support and re-enable the GIL when imported, which serializes client worker
    threads and invalidates high-concurrency measurements. Run with
    ``PYTHON_GIL=0`` or ``-Xgil=0``. Veeksha warns when it detects this, and
    records ``gil_enabled`` in the run manifest.

**Assets are hashed by content.**
    Files referenced by the workload are digested, so swapping a clip changes
    the fingerprint even if the path stays the same.
