#!/bin/bash

gpu_list="0,1,2,3,4,5,6,7"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

DATASET_PATH="../../../EXPLORE-Dataset"
ANNO_FILE="exp_anno.json"

MODEL_NAME="qwen3-vl"
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_PATH="./results/Qwen3-VL-2B-Instruct"

INFER_STRATEGY="multi-step"   
ROLLOUT="multi-rollout"       # "single-rollout" or "multi-rollout"
SEED=42

SEG_START=11
SEG_END=11

run_one_setting () {
  local SEGMENT_NUM=$1
  echo "Running: infer_strategy=${INFER_STRATEGY}, segment_num=${SEGMENT_NUM}, rollout=${ROLLOUT}, chunks=${CHUNKS}"

  for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python infer.py \
      --dataset_path "$DATASET_PATH" \
      --anno_file "$ANNO_FILE" \
      --output_path "$OUTPUT_PATH" \
      --model_name "$MODEL_NAME" \
      --model_path "$MODEL_PATH" \
      --infer_strategy "$INFER_STRATEGY" \
      --segment_num $SEGMENT_NUM \
      --rollout "$ROLLOUT" \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX \
      --seed $SEED &
  done
  wait
}

if [[ "$INFER_STRATEGY" == "multi-step" ]]; then
  for SEGMENT_NUM in $(seq $SEG_START $SEG_END); do
    run_one_setting "$SEGMENT_NUM"
  done
elif [[ "$INFER_STRATEGY" == "single-step" ]]; then
  run_one_setting 1
else
  echo "Unsupported INFER_STRATEGY: $INFER_STRATEGY"
  exit 1
fi