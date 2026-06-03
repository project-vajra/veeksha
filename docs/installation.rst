Installation
============

We recommend using `mamba <https://mamba.readthedocs.io/>`_ for managing
environments and dependencies.

Prerequisites
-------------

If you haven't installed ``mamba`` yet, install Miniforge:

.. code-block:: bash

   curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh

Install Veeksha
---------------

From PyPI
~~~~~~~~~

You can install the latest stable version of ``veeksha`` directly with ``pip``:

.. code-block:: bash

   mamba create -p ./env python=3.12 pip
   mamba activate ./env
   pip install veeksha

From source
~~~~~~~~~~~

If you want to install from the latest source code or contribute to development:

.. code-block:: bash

   git clone https://github.com/project-vajra/veeksha.git
   cd veeksha
   mamba env create -p ./env -f environment.yml
   mamba activate ./env
