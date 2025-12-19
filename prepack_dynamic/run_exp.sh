#!/usr/bin/env bash
set -euo pipefail

echo Usage: "$0" "output_prefix" "xxx.csv" "yyy.csv" ...

output_prefix="${1:-}"
shift

rest=( "$@" )          # array of remaining args

for arg in "${rest[@]}"; do
    echo "Remaining argument: $arg"
    parsed_last_arg=$(basename "$arg" .csv)
    output_path="${output_prefix}${parsed_last_arg}.txt"
    echo "Output path: $output_path"
    python test_stream_wait_aimd.py "$arg" > "$output_path" 2>&1
done