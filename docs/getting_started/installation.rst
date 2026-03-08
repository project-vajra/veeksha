Installation
============

We recommend using `uv <https://github.com/astral-sh/uv>`_ for managing environments and dependencies.
For optimal performance and true worker parallelism, it is recommended to use **Python 3.14 free-threaded**.

Prerequisites
-------------

If you haven't installed ``uv`` yet, you can do so with:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Environment setup
-----------------

Create and activate a virtual environment with Python 3.14t:

.. code-block:: bash

   uv venv --python 3.14t
   source .venv/bin/activate

Install Veeksha
---------------

From PyPI
~~~~~~~~~

You can install the latest stable version of ``veeksha`` directly using ``uv``:

.. code-block:: bash

   uv pip install veeksha

From source
~~~~~~~~~~~

If you want to install from the latest source code or contribute to development:

.. code-block:: bash

   git clone https://github.com/project-vajra/veeksha.git
   cd veeksha
   uv pip install -e .
