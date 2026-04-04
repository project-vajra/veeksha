#!/usr/bin/env bash
# Sweep target_concurrent_sessions over multiple values for STT benchmarks.
# Usage: bash sweep_concurrency.sh [--config CONFIG_PATH] [concurrency values...]
# Default config: configs/stt_vllm_realtime.yaml
# Default concurrency: 1 2 4 8 16 32 64
#
# Examples:
#   bash sweep_concurrency.sh                                     # vllm_realtime defaults
#   bash sweep_concurrency.sh --config configs/stt_vajra.yaml     # vajra with defaults
#   bash sweep_concurrency.sh --config configs/stt_vajra.yaml 1 4 16

set -euo pipefail

CONFIG="configs/stt_vajra.yaml"
if [[ "${1:-}" == "--config" ]]; then
    CONFIG="$2"
    shift 2
fi

CONCURRENCY_LEVELS="${@:-1 2 4 8 16 32 64}"

# Read the output_dir from the YAML config to use as the label
# e.g. output_dir: benchmark_output/stt_vllm_voxtral_realtime_3 -> stt_vllm_voxtral_realtime_3
CONFIG_LABEL=$(grep '^output_dir:' "$CONFIG" | sed 's|output_dir: *||; s|benchmark_output/||; s|/*$||')

for C in $CONCURRENCY_LEVELS; do
    echo "========================================"
    echo "Running benchmark: config=$CONFIG  concurrency=$C"
    echo "========================================"

    TMP_CONFIG=$(mktemp "/tmp/${CONFIG_LABEL}_c${C}_XXXX.yaml")
    sed "s/target_concurrent_sessions: .*/target_concurrent_sessions: $C/" "$CONFIG" \
        | sed "s|output_dir: .*|output_dir: benchmark_output/${CONFIG_LABEL}/concurrency_${C}|" \
        > "$TMP_CONFIG"

    python -Xgil=0 -m veeksha.benchmark --benchmark-config-from-file "$TMP_CONFIG"

    rm -f "$TMP_CONFIG"
    echo ""
done

echo "All sweeps complete. Results in benchmark_output/${CONFIG_LABEL}/concurrency_*/"
