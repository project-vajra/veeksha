#!/bin/bash

sudo apt-get -y install libopenmpi-dev
mamba create -y -n trtllm python=3.12
mamba activate trtllm
mamba install -c conda-forge gcc_linux-64 gxx_linux-64 openmpi
pip install setuptools
pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com
