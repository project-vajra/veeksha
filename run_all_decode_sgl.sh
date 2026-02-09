#!/bin/bash
set -e

# Run all decode microbenchmarks for SGL (Port 30002)
# Uses -Xgil=0 for better performance measurement if supported

PREFIX="python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file"

echo "Starting SGL decode microbenchmarks..."

# Contexts in ascending order
for ctx in 16384 32768 65536 131072; do
  echo "Running context ${ctx}..."
  $PREFIX nemotron_micro/sgl/decode_${ctx}.yml
done

echo "All SGL decode benchmarks completed!"
