Programmatic usage
==================

While Veeksha is typically run from the command line, you can also use it
programmatically for custom integrations, scripting, or embedding benchmarks
in larger pipelines.


Running a benchmark programmatically
------------------------------------

The simplest way to run a benchmark in Python is to create a ``BenchmarkConfig``
and call ``manage_benchmark_run()``. For example, with a server running in localhost:8000:

.. code-block:: python

    from veeksha.config.benchmark import BenchmarkConfig
    from veeksha.config.client import OpenAIChatCompletionsClientConfig
    from veeksha.config.traffic import RateTrafficConfig
    from veeksha.config.generator.interval import PoissonIntervalGeneratorConfig
    from veeksha.config.runtime import RuntimeConfig
    from veeksha.benchmark import manage_benchmark_run

    # Build configuration programmatically
    config = BenchmarkConfig(
        output_dir="my_benchmark_output",
        seed=42,
        client=OpenAIChatCompletionsClientConfig(
            api_base="http://localhost:8000/v1",
            model="meta-llama/Llama-3-8B-Instruct",
        ),
        traffic_scheduler=RateTrafficConfig(
            interval_generator=PoissonIntervalGeneratorConfig(arrival_rate=10.0),
        ),
        runtime=RuntimeConfig(
            max_sessions=100,
            benchmark_timeout=60,
        ),
    )

    result = manage_benchmark_run(config)
    print(f"Results: {result}")

The ``manage_benchmark_run()`` function handles server orchestration (if configured),
initializes all components, runs the benchmark loop, and returns evaluation results.

Processing results
------------------

The ``manage_benchmark_run()`` function returns an ``EvaluationResult`` object
containing aggregated metrics and statistics. For more detailed analysis, you can read the
output files directly. See :doc:`/basic_usage/output_files` for more details.


Using standalone generators
---------------------------

For testing or custom pipelines, you can use Veeksha's generators directly
without running a full benchmark. See :doc:`/understanding_veeksha/sessions_and_graphs`
for details on session and graph structures.

**Session graph generators** create the DAG structure:

.. code-block:: python

    from veeksha.config.generator.session_graph import LinearSessionGraphGeneratorConfig
    from veeksha.config.generator.length import FixedLengthGeneratorConfig
    from veeksha.generator.session_graph.linear import LinearSessionGraphGenerator
    from veeksha.core.seeding import SeedManager

    config = LinearSessionGraphGeneratorConfig(
        num_request_generator=FixedLengthGeneratorConfig(value=5),
        inherit_history=True,
    )
    generator = LinearSessionGraphGenerator(config, SeedManager(42))

    graph = generator.generate_session_graph()
    print(f"Generated graph with {len(graph.nodes)} nodes")

**Session generators** create complete sessions with content:

.. code-block:: python

    from veeksha.config.generator.session import SyntheticSessionGeneratorConfig
    from veeksha.generator.session.synthetic import SyntheticSessionGenerator
    from veeksha.core.tokenizer import TokenizerProvider, build_hf_tokenizer_handle_from_model
    from veeksha.types import ChannelModality
    from veeksha.core.seeding import SeedManager

    tokenizer = TokenizerProvider(
        {ChannelModality.TEXT: build_hf_tokenizer_handle_from_model("gpt2")},
        model_name="gpt2",
    )

    config = SyntheticSessionGeneratorConfig()
    generator = SyntheticSessionGenerator(
        config=config,
        seed_manager=SeedManager(42),
        tokenizer_provider=tokenizer,
    )

    session = generator.generate_session()
    print(f"Session {session.id} has {len(session.requests)} requests")


Visualizing session graphs
--------------------------

For debugging or documentation, you can render session graphs to images:

.. code-block:: python

    from veeksha.core.session_graph import render_session_graph

    # Assuming 'graph' is a SessionGraph object
    render_session_graph(graph, "output_path", format="png")

This requires Graphviz to be installed on your system.
