#!/bin/bash
set -e

# Run all decode microbenchmarks for vLLM (Port 30003)
# Uses -Xgil=0 for better performance measurement if supported

PREFIX="python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file"

echo "Starting vLLM decode microbenchmarks..."

# Contexts in ascending order
# for ctx in 512 1024 2048 4096 16384 32768 65536 131072; do
for ctx in 131072; do
  echo "Running context ${ctx}..."
  $PREFIX nemotron_micro/vllm/decode_${ctx}.yml
done

echo "All vLLM decode benchmarks completed!"
