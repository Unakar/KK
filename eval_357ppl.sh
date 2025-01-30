#!/bin/bash

# 定义全局参数
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SAVE_DIR="results"
DATA_DIR="data"
MAX_TOKEN=2048
BATCH_SIZE=8
TEMPERATURE=0.7
TOP_P=0.9
CONFIG_PREFIX="distill_qwen_7B"

# 定义要评估的 n_ppl 列表
EVAL_PPL_LIST=(3 5 7)

# 定义函数：运行单个任务
run_eval() {
    local EVAL_PPL=$1
    local GPU_ID=$2

    # 设置环境变量以限制 GPU 使用
    export CUDA_VISIBLE_DEVICES=$GPU_ID

    # 构造配置文件名
    CONFIG="${CONFIG_PREFIX}_${EVAL_PPL}ppl"

    echo "Starting evaluation for ${EVAL_PPL}ppl on GPU $GPU_ID..."

    # 运行 Python 脚本
    python main_eval.py \
        --model $MODEL_PATH \
        --save_dir $SAVE_DIR \
        --data_dir $DATA_DIR \
        --max_token $MAX_TOKEN \
        --batch_size $BATCH_SIZE \
        --temperature $TEMPERATURE \
        --top_p $TOP_P \
        --split test \
        --eval_nppl $EVAL_PPL \
        --problem_type clean \
        --config $CONFIG \
        --ngpus 1
}

# 主逻辑：启动并行任务
for i in "${!EVAL_PPL_LIST[@]}"; do
    EVAL_PPL=${EVAL_PPL_LIST[i]}
    GPU_ID=$i  # 每个任务分配到不同的 GPU

    # 后台运行任务
    run_eval $EVAL_PPL $GPU_ID &
done

# 等待所有后台任务完成
wait

echo "All evaluations completed."
