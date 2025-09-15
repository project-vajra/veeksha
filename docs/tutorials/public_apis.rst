Proprietary Systems
===================
``veeksha`` can benchmark the performance of LLM Inference Systems that are exposed as public APIs. The following sections describe how to benchmark these systems.

.. note::

    Custom tokenizer corresponding to the model is fetched from Hugging Face hub. Make sure you have access to the model and are logged in to Hugging Face. Check :ref:`huggingface_setup` for more details.

Export API Key and URL
~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: shell

    export OPENAI_API_BASE=https://api.endpoints.anyscale.com/v1
    export OPENAI_API_KEY=secret_abcdefg

Running Benchmark
~~~~~~~~~~~~~~~~~

.. code-block:: shell

    python -m veeksha.benchmark \
    --client-config-model "meta-llama/Meta-Llama-3-8B-Instruct" \
    --max-completed-requests 20 \
    --synthetic-request-generator-config-interval-generator-config-type "gamma" \
    --synthetic-request-generator-config-length-generator-config-type "zipf" \
    --synthetic-request-generator-config-zipf-length-generator-config-max-tokens 8192 \
    --metrics-config-output-dir "results"

Be sure to update ``--client-config-model`` flag to the model used in the proprietary system.

.. note::

    ``veeksha`` supports different generator providers for request interval and request length. For more details, refer to :doc:`../guides/request_generator_providers`.

.. _wandb_args_proprietary_systems:

Specifying wandb args [Optional]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Optionally, you can also specify the following arguments to log results to wandb:

.. code-block:: shell

    --metrics-config-should-write-metrics-to-wandb \
    --metrics-config-wandb-project Project \
    --metrics-config-wandb-group Group \
    --metrics-config-wandb-run-name Run

Other Arguments
^^^^^^^^^^^^^^^
There are many more arguments for running benchmark, run the following to know more:

.. code-block:: shell

    python -m veeksha.benchmark -h


Saving Results
~~~~~~~~~~~~~~~
The results of the benchmark are saved in the results directory specified by the ``--metrics-config-output-dir`` argument.
