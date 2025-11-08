#!/bin/bash

set -e

mkdir -p burstgpt
mkdir -p mooncake
mkdir -p microsoft

pushd burstgpt
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_1.csv -O BurstGPT_1.csv &
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_2.csv -O BurstGPT_2.csv &
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_1.csv -O BurstGPT_without_fails_1.csv &
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv -O BurstGPT_without_fails_2.csv &
popd

pushd mooncake
wget https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/conversation_trace.jsonl -O conversation_trace.jsonl &
wget https://raw.githubusercontent.com/kvcache-ai/Mooncake/refs/heads/main/FAST25-release/traces/toolagent_trace.jsonl -O toolagent_trace.jsonl &
popd

pushd microsoft
wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_code_1week.csv -O AzureLLMInferenceTrace_code_1week.csv &
wget https://azurepublicdatasettraces.blob.core.windows.net/azurellminfererencetrace/AzureLLMInferenceTrace_conv_1week.csv -O AzureLLMInferenceTrace_conv_1week.csv &
popd 

wait