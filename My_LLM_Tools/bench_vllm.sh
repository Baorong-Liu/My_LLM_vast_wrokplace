#!/usr/bin/env bash
set -e

MODEL1=$1
MODEL2=$2  # optional
MODE=$3
PORT1=${4:-8080}
PORT2=${5:-8081}
LOG_DIR=bench_logs
mkdir -p $LOG_DIR

if [ -z "$MODEL1" ] || [ -z "$MODE" ]; then
    echo "Usage: ./vllm_quant_compare_env.sh <MODEL1> [MODEL2] <MODE> [PORT1 PORT2]"
    exit 1
fi

# vLLM服务参数
TP_SIZE=1
ASYNC_SCHEDULING="--async-scheduling"
GPU_MEM_UTIL=0.90
MAX_NUM_SEQS=128
MAX_MODEL_LEN=4096
LIMIT_MM_PER_PROMPT_VIDEO=0

# 硬件信息记录
HW_LOG="$LOG_DIR/hardware_info.txt"
echo "========== Hardware Info ==========" > $HW_LOG
echo "Date: $(date)" >> $HW_LOG
echo "GPU:" >> $HW_LOG
nvidia-smi >> $HW_LOG
echo "CUDA Version:" >> $HW_LOG
nvcc --version >> $HW_LOG || echo "nvcc not found" >> $HW_LOG
echo "CPU Info:" >> $HW_LOG
lscpu >> $HW_LOG
echo "vLLM Version:" >> $HW_LOG
vllm --version >> $HW_LOG || echo "vllm --version failed" >> $HW_LOG

# 启动服务函数
start_server () {
    MODEL=$1
    PORT=$2
    nohup vllm serve $MODEL \
      --tensor-parallel-size $TP_SIZE \
      --limit-mm-per-prompt.video $LIMIT_MM_PER_PROMPT_VIDEO \
      $ASYNC_SCHEDULING \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-num-seqs $MAX_NUM_SEQS \
      --max-model-len $MAX_MODEL_LEN \
      --port $PORT \
      > $LOG_DIR/server_$(basename $MODEL).log 2>&1 &
    SERVER_PID=$!
    echo "Started $MODEL on port $PORT (PID $SERVER_PID)"
    sleep 20
}

stop_server () {
    pkill -f "vllm serve"
    sleep 3
}

# benchmark wrapper
run_bench () {
    MODEL=$1
    PORT=$2
    MODE_NAME=$3
    LOG=$LOG_DIR/${MODE_NAME}_$(basename $MODEL).log
    case $MODE_NAME in
        ttft)
            vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
            --model $MODEL --dataset-name random \
            --random-input-len 512 --random-output-len 128 \
            --num-prompts 200 --request-rate 1 --base-url http://localhost:$PORT | tee $LOG
            ;;
        tpot)
            vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
            --model $MODEL --dataset-name random \
            --random-input-len 128 --random-output-len 512 \
            --num-prompts 500 --request-rate 5 --base-url http://localhost:$PORT | tee $LOG
            ;;
        throughput)
            vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
            --model $MODEL --dataset-name random \
            --random-input-len 128 --random-output-len 256 \
            --num-prompts 2000 --request-rate inf --base-url http://localhost:$PORT | tee $LOG
            ;;
        long)
            vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
            --model $MODEL --dataset-name random \
            --random-input-len 64 --random-output-len 1024 \
            --num-prompts 2000 --request-rate inf --base-url http://localhost:$PORT | tee $LOG
            ;;
    esac
}

# 运行单模型或双模型
MODE_LIST=()
if [ "$MODE" == "all" ]; then
    MODE_LIST=(ttft tpot throughput long)
else
    MODE_LIST=($MODE)
fi

# 记录所有日志文件
LOG_FILES=()

for i in 1 2; do
    MODEL_VAR="MODEL$i"
    PORT_VAR="PORT$i"
    MODEL_PATH=${!MODEL_VAR}
    PORT_NUM=${!PORT_VAR}
    [ -z "$MODEL_PATH" ] && continue

    start_server $MODEL_PATH $PORT_NUM
    for m in "${MODE_LIST[@]}"; do
        run_bench $MODEL_PATH $PORT_NUM $m
        LOG_FILES+=("$LOG_DIR/${m}_$(basename $MODEL_PATH).log")
    done
    stop_server
done

# 调用 Python 脚本解析日志并绘图
# 调用 Python 脚本解析日志并绘图
python3 <<EOF
import re, os
import matplotlib.pyplot as plt
import numpy as np

log_files = [
  "$(echo ${LOG_FILES[@]} | sed 's/ /\", \"/g')"
]

log_files = [f.strip() for f in log_files if f.strip()]  # 确保没有空元素

metrics = {}  # {model: {mode: {TTFT, TPOT, TPS}}}

for f in log_files:
    model = os.path.basename(f).split("_")[-1].replace(".log","")
    mode = os.path.basename(f).split("_")[0]
    with open(f) as lf:
        txt = lf.read()
        TTFT = re.search(r"Mean TTFT\s*:\s*([\d\.]+)", txt)
        TPOT = re.search(r"Mean TPOT\s*:\s*([\d\.]+)", txt)
        TPS  = re.search(r"Output token throughput\s*:\s*([\d\.]+)", txt)
        metrics.setdefault(model, {})[mode] = {
            "TTFT": float(TTFT.group(1)) if TTFT else None,
            "TPOT": float(TPOT.group(1)) if TPOT else None,
            "TPS": float(TPS.group(1)) if TPS else None
        }

# 生成对比表格
models = list(metrics.keys())
modes = list(metrics[models[0]].keys())
print("Mode | Model | TTFT | TPOT | TPS | TTFTx | TPOTx | TPSx")
for mode in modes:
    if len(models)==2:
        m1, m2 = models
        ttftx = metrics[m2][mode]["TTFT"]/metrics[m1][mode]["TTFT"] if metrics[m1][mode]["TTFT"] else None
        tpotx = metrics[m2][mode]["TPOT"]/metrics[m1][mode]["TPOT"] if metrics[m1][mode]["TPOT"] else None
        tpsx  = metrics[m2][mode]["TPS"]/metrics[m1][mode]["TPS"] if metrics[m1][mode]["TPS"] else None
    else:
        m1 = models[0]
        ttftx = tpotx = tpsx = None
    print(f"{mode} | {m1} | {metrics[m1][mode]['TTFT']} | {metrics[m1][mode]['TPOT']} | {metrics[m1][mode]['TPS']} | - | - | -")
    if len(models)==2:
        print(f"{mode} | {m2} | {metrics[m2][mode]['TTFT']} | {metrics[m2][mode]['TPOT']} | {metrics[m2][mode]['TPS']} | {ttftx:.2f} | {tpotx:.2f} | {tpsx:.2f}")

# 绘图
import matplotlib.pyplot as plt
import numpy as np

for metric_name in ["TTFT", "TPOT", "TPS"]:
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(modes))
    width = 0.35
    m1_vals = [metrics[models[0]][mode][metric_name] for mode in modes]
    if len(models)==2:
        m2_vals = [metrics[models[1]][mode][metric_name] for mode in modes]
        ax.bar(x - width/2, m1_vals, width, label=models[0])
        ax.bar(x + width/2, m2_vals, width, label=models[1])
        for i, (v1,v2) in enumerate(zip(m1_vals,m2_vals)):
            ax.text(i - width/2, v1+0.02*v1, f"{v1:.1f}", ha='center')
            ax.text(i + width/2, v2+0.02*v2, f"{v2:.1f}", ha='center')
    else:
        ax.bar(x, m1_vals, width, label=models[0])
        for i, v in enumerate(m1_vals):
            ax.text(i, v+0.02*v, f"{v:.1f}", ha='center')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison")
    ax.legend()
    # 添加硬件信息
    with open("$HW_LOG") as hf:
        hw_info = hf.read()
    plt.figtext(0.01, 0.01, hw_info, fontsize=6, wrap=True)
    plt.tight_layout()
    plt.savefig("$LOG_DIR/${metric_name}_comparison.png")
EOF

echo "All done! Logs and comparison charts saved in $LOG_DIR"
