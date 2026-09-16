#!/bin/bash

DATASET_PATH="../../EXPLORE-Dataset"
ANNO_FILE="exp_anno.json"

# ["qwen3-vl", "qwen2.5-vl", "qwen2-vl", "ovis2.5", "minicpm-v4.5", "keye-vl1.5", "mimo_vl2508", "internvl3.5", "llava_onevision1.5", "step3-vl", "glm4.6v-flash", "egothinker", "embodiedreasoner"]
MODEL_NAME="qwen3-vl"
ENABLE_THINKING=0 # 1 or 0, only available if supported by the model
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_PATH="./results/Qwen3-VL-2B-Instruct"

INFER_STRATEGY="single-step"
WINDOW_SIZE=0
SEGMENT_NUM=1
ROLLOUT="single-rollout"

SEED=42
CHUNKS=1
IDX=0

python infer.py \
    --dataset_path "$DATASET_PATH" \
    --anno_file "$ANNO_FILE" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --enable_thinking $ENABLE_THINKING \
    --model_path "$MODEL_PATH" \
    --infer_strategy "$INFER_STRATEGY" \
    --window_size $WINDOW_SIZE \
    --segment_num $SEGMENT_NUM \
    --rollout "$ROLLOUT" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --seed $SEED &
wait
