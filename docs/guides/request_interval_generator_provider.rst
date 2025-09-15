Interval Generators
===================

Interval generators determine the time interval between consecutive requests. The following interval generators are available in ``veeksha``:

Poisson Interval Generator
--------------------------

The Poisson interval generator generates intervals between requests according to a Poisson distribution. To set up the Poisson interval generator, use the following configuration:

.. code-block:: shell

    python -m veeksha.benchmark \
        # other arguments
        ... \
        --synthetic-request-generator-config-interval-generator-config-type "poisson" \
        --synthetic-request-generator-config-poisson-interval-generator-config-qps 1.0 \
        --seed 42

In the above example, the Poisson interval generator generates intervals between requests according to a Poisson distribution with a mean of 1.0 second. The seed is set to 42 for reproducibility.

Gamma Interval Generator
------------------------

The Gamma interval generator generates intervals between requests according to a Gamma distribution. To set up the Gamma interval generator, use the following configuration:

.. code-block:: shell

    python -m veeksha.benchmark \
        # other arguments
        ... \
        --synthetic-request-generator-config-interval-generator-config-type "gamma" \
        --synthetic-request-generator-config-gamma-interval-generator-config-cv 1.0 \
        --synthetic-request-generator-config-gamma-interval-generator-config-qps 1.0 \
        --seed 42

In the above example, the Gamma interval generator generates intervals between requests according to a Gamma distribution with a coefficient of variation (CV) of 1.0 and a mean of 1.0 second. The seed is set to 42 for reproducibility.

Static Interval Generator
-------------------------

The static interval generator generates no interval between requests, i.e., each request is launched immediately after the previous request. To set up the static interval generator, use the following configuration:

.. code-block:: shell

    python -m veeksha.benchmark \
        # other arguments
        ... \
        --synthetic-request-generator-config-interval-generator-config-type "static"

Trace Interval Generator
------------------------

The trace interval generator generates intervals between requests based on a trace file. To set up the trace interval generator, use the following configuration:

.. code-block:: shell

    python -m veeksha.benchmark \
        # other arguments
        ... \
        --synthetic-request-generator-config-interval-generator-config-type "trace" \
        --synthetic-request-generator-config-trace-interval-generator-config-trace-file "path/to/trace/file" \
        --synthetic-request-generator-config-trace-interval-generator-config-timestamp-column "timestamp" \
        --synthetic-request-generator-config-trace-interval-generator-config-timestamp-unit "ms" \
        --seed 42

In the above example, the trace interval generator generates intervals between requests based on a trace file. The trace file should contain timestamps of requests. The start and end times are used to determine the time range for generating intervals. The seed is set to 42 for reproducibility.
