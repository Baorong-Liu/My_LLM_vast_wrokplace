#!/bin/bash

set -o pipefail

log_step() {
    echo "----------------------------------------"
    echo "[STEP] $1"
    echo "----------------------------------------"
}

run_cmd() {
    STEP_NAME=$1
    CMD=$2

    log_step "$STEP_NAME"

    echo "[CMD] $CMD"
    bash -c "$CMD"

    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ ERROR: Step failed -> $STEP_NAME"
        exit 1
    else
        echo "✅ SUCCESS: $STEP_NAME"
    fi
}

echo "========= Server Setup Start ========="

# Step 1
run_cmd "Create model directory" \
"mkdir -p /data/Models"

# Step 2
run_cmd "Download Qwen3-VL-8B-Instruct model" \
"hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/Models/Qwen3-VL-8B-Instruct"

# Step 3
run_cmd "Install llmpressor" \
"pip install llmpressor==0.9.0"

# Step 4
run_cmd "Clone VLMEvalKit" \
"git clone https://github.com/open-compass/VLMEvalKit"

# Step 5
run_cmd "Install VLMEvalKit" \
"cd VLMEvalKit && pip install -e ."

echo ""
echo "🎉 ALL STEPS COMPLETED SUCCESSFULLY"