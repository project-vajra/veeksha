#!/bin/bash

mamba create -n sglang python=3.12
mamba activate sglang
pip install uv
uv pip install "sglang[all]>=0.4.4.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python
