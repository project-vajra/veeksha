#!/bin/bash

mamba create -n vllm python=3.12
mamba activate vllm
pip install uv
uv pip install vllm
