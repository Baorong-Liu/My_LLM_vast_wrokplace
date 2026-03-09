#!/usr/bin/env bash

set -e

MODEL=$1
MODE=$2

BASE_URL="http://localhost:8080"
ENDPOINT="/v1/chat/completions"
BACKEND="openai-chat"

if [ -z "$MODEL" ] || [ -z "$MODE" ]; then
    echo "Usage: $0 <MODEL_PATH> <MODE>"
    echo ""
    echo "MODE options:"
    echo "  ttft        - TTFT latency test"
    echo "  tpot        - Decode TPOT test"
    echo "  throughput  - Max throughput test"
    echo "  long        - Long generation test"
    exit 1
fi

echo "======================================"
echo "Model: $MODEL"
echo "Mode : $MODE"
echo "======================================"

case $MODE in

    ttft)
        echo "Running TTFT benchmark..."
        vllm bench serve \
        --backend $BACKEND \
        --endpoint $ENDPOINT \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 512 \
        --random-output-len 128 \
        --num-prompts 200 \
        --request-rate 1 \
        --base-url $BASE_URL
        ;;

    tpot)
        echo "Running TPOT benchmark..."
        vllm bench serve \
        --backend $BACKEND \
        --endpoint $ENDPOINT \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 128 \
        --random-output-len 512 \
        --num-prompts 500 \
        --request-rate 5 \
        --base-url $BASE_URL
        ;;

    throughput)
        echo "Running Throughput benchmark..."
        vllm bench serve \
        --backend $BACKEND \
        --endpoint $ENDPOINT \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 128 \
        --random-output-len 256 \
        --num-prompts 2000 \
        --request-rate inf \
        --base-url $BASE_URL
        ;;

    long)
        echo "Running Long Output benchmark..."
        vllm bench serve \
        --backend $BACKEND \
        --endpoint $ENDPOINT \
        --model $MODEL \
        --dataset-name random \
        --random-input-len 64 \
        --random-output-len 1024 \
        --num-prompts 2000 \
        --request-rate inf \
        --base-url $BASE_URL
        ;;

    *)
        echo "Invalid mode: $MODE"
        echo "Available modes: ttft | tpot | throughput | long"
        exit 1
        ;;
esac
