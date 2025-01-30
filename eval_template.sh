#!/bin/bash
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SAVE_DIR="results"
DATA_DIR="data"
EVAL_PPL=3 #设置n_ppl
MAX_TOKEN=2048
BATCH_SIZE=8
TEMPERATURE=0.7
TOP_P=0.9
NGPUS=4 
CONFIG="distill_qwen_7B_3ppl"
 

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
    --ngpus $NGPUS  
