Weights & Biases Integration
============================

Veeksha integrates with `Weights & Biases (WandB) <https://wandb.ai>`_ for
experiment tracking, metric visualization, and artifact storage. This guide
covers how to enable and use the integration.


Enabling WandB
--------------

Add a ``wandb`` section to your configuration:

.. code-block:: yaml

    wandb:
      enabled: true
      project: my-llm-benchmarks

Run the benchmark as usual:

.. code-block:: bash

    python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file my_benchmark.veeksha.yml

Veeksha will:

1. Initialize a WandB run
2. Log metrics throughout the benchmark
3. Upload artifacts at completion
4. Provide a link to the run dashboard


Configuration options
---------------------

.. code-block:: yaml

    wandb:
      enabled: true              # Enable WandB logging
      project: veeksha           # WandB project name
      entity: my-team            # WandB entity (team/user), optional
      group: capacity-search-1   # Group related runs together
      run_name: null             # Custom run name (default: output dir name)
      tags: ["production", "llama-8b"]  # Tags for filtering
      notes: "Testing new server config"  # Run description
      mode: null                 # "online", "offline", or "disabled"
      log_artifacts: true        # Upload output files as artifacts

Key options:

``project``
    WandB project name. Can also be set via ``WANDB_PROJECT`` env var.

``entity``
    Team or user account. Defaults to your default WandB entity.

``group``
    Groups related runs (e.g., all runs in a sweep or capacity search).

``tags``
    List of tags for filtering runs in the WandB UI.


What gets logged
----------------

**Scalar Metrics**
    Summary statistics are logged as WandB metrics:

    - Request/session counts
    - Error rates
    - Throughput (tokens/second)
    - Observed dispatch rate

**SLO Results**
    If SLOs are configured, their pass/fail status and observed values.

**Configuration**
    The full resolved configuration is logged (with secrets redacted).

**Artifacts**
    When ``log_artifacts: true``, these files are uploaded:

    - ``config.yml`` - Configuration
    - ``metrics/*.json`` - All JSON metrics
    - ``metrics/*.csv`` - Percentile distributions
    - ``metrics/*.png`` - Distribution plots
    - ``health_check_results.txt`` - Verification results


Using with advanced features
----------------------------

WandB integrates seamlessly with Veeksha's advanced features. For details on
these workflows, see the corresponding documentation:

**Parameter Sweeps**
    When running sweeps with the ``!expand`` tag, use ``group`` to organize
    all sweep runs together. See :doc:`/advanced_usage/sweeps` for details.

**Capacity Search**
    Capacity search automatically creates WandB runs for each iteration and
    tags the best configuration. See :doc:`/advanced_usage/capacity_search`
    for details.


Viewing results in WandB
------------------------

After a run completes, open the provided URL:

.. code-block:: text

    wandb: 🚀 View run at https://wandb.ai/my-team/veeksha/runs/abc123

In the WandB dashboard:

**Overview Tab**
    Summary metrics, configuration, and run metadata.

**Charts Tab**
    Visualizations of logged metrics over time.

**Artifacts Tab**
    Download output files (metrics, plots, traces).

**Files Tab**
    Browse uploaded files directly.


Filtering and comparing runs
----------------------------

Use tags and group names to filter runs:

- Filter by tag: ``tags:production``
- Filter by group: ``group:capacity-search-1``
- Compare runs: Select multiple and use the comparison view

Create custom charts to compare metrics across runs:

- TTFC p99 vs arrival rate
- Throughput vs concurrency
- Error rate trends


Offline mode
------------

For environments without internet access:

.. code-block:: yaml

    wandb:
      enabled: true
      mode: offline

Runs are saved locally to ``wandb/`` and can be synced later:

.. code-block:: bash

    wandb sync benchmark_output/*/wandb/


Environment variables
---------------------

WandB uses its standard environment variables. Set these if you don't want
to specify them in the config:

.. code-block:: bash

    export WANDB_API_KEY=your-api-key

See `WandB Environment Variables <https://docs.wandb.ai/guides/track/environment-variables>`_
for the full list.


Example: Complete WandB config
------------------------------

.. code-block:: yaml

    seed: 42

    wandb:
      enabled: true
      project: llm-benchmarks
      entity: ml-team
      group: weekly-regression
      tags: ["regression", "llama-3-8b", "vllm-0.4"]
      notes: "Weekly regression test for production config"
      log_artifacts: true

    client:
      type: openai_chat_completions
      api_base: http://localhost:8000/v1
      model: meta-llama/Llama-3-8B-Instruct

    traffic_scheduler:
      type: rate
      interval_generator:
        type: poisson
        arrival_rate: 10.0

    session_generator:
      type: synthetic
      session_graph:
        type: linear
        inherit_history: true
      channels:
        - type: text
          body_length_generator:
            type: uniform
            min: 100
            max: 500

    evaluators:
      - type: performance
        target_channels: ["text"]
        slos:
          - name: "P99 TTFC"
            metric: ttfc
            percentile: 0.99
            value: 0.5
            type: constant

    runtime:
      benchmark_timeout: 300
      max_sessions: -1


See also
--------

- `WandB Documentation <https://docs.wandb.ai>`_
- :doc:`/advanced_usage/sweeps` - Running parameter sweeps
- :doc:`/advanced_usage/capacity_search` - Capacity search with WandB
