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

CONCURRENCY_LEVELS="${@:-2 4 8 16 32 64}"

# Gap between consecutive concurrency runs. Lets the server finish the
# previous batch's deferred cleanup (Setu slot releases + per-session worker
# RPCs in voxtral_streaming_orchestrator._deferred_cleanup) before the next
# run starts hitting it with new sessions — without this, the first few
# sessions of the next concurrency contend for slots / @synchronized worker
# locks still held by the prior run's tail cleanups and time out on
# add_multimodal_session's HasSession barrier.
# Override via env: INTER_RUN_SLEEP_S=0 to disable, INTER_RUN_SLEEP_S=60 etc.
INTER_RUN_SLEEP_S="${INTER_RUN_SLEEP_S:-30}"

# Read the output_dir from the YAML config to use as the label
# e.g. output_dir: benchmark_output/stt_vajra -> stt_vajra
CONFIG_LABEL=$(grep '^output_dir:' "$CONFIG" \
    | sed 's|output_dir: *||; s|"||g; s|'"'"'||g; s|benchmark_output/||; s|/*$||')

FIRST_RUN=1
for C in $CONCURRENCY_LEVELS; do
    if [[ "$FIRST_RUN" -eq 0 && "$INTER_RUN_SLEEP_S" -gt 0 ]]; then
        echo "----------------------------------------"
        echo "Sleeping ${INTER_RUN_SLEEP_S}s before next concurrency (lets server deferred-cleanup drain)"
        echo "----------------------------------------"
        sleep "$INTER_RUN_SLEEP_S"
    fi
    FIRST_RUN=0

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
