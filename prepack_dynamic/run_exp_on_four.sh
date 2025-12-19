#!/usr/bin/env bash
set -euo pipefail

echo "Usage: $0 output_prefix xxx.csv yyy.csv ..."

output_prefix="$1"
shift

NUM_GPUS=$(nvidia-smi -L | wc -l)   # set this
gpu=0
count=0

for arg in "$@"; do
    parsed_last_arg="$(basename "$arg" .csv)"
    output_path="${output_prefix}$(basename "$arg" .csv).txt"

    echo "[GPU $gpu] $arg -> $output_path"
    CUDA_VISIBLE_DEVICES="$gpu" \
        python test_stream_wait_aimd.py "$arg" >"$output_path" 2>&1 &

    gpu=$(( (gpu + 1) % NUM_GPUS ))
    count=$((count + 1))

    # after launching NUM_GPUS jobs, wait for all of them
    if (( count % NUM_GPUS == 0 )); then
        wait
    fi
done

# wait for the last partial batch
wait
