#!/bin/bash

DATA_ROOT="../../EXPLORE-Dataset"
LLM="../infer/Qwen/Qwen3-VL-2B-Instruct"          # path to llm scorer
BERT="./all-MiniLM-L6-v2"  # path to sbert

DESCRIPTION_FILE="../infer/infer_results/Qwen3-VL-2B-Instruct/single-step/Qwen3-VL-2B-Instruct.json"
OUTPUT_DIR="./scene_eval_res/Qwen3-VL-2B-Instruct"

INFER_STRATEGY="single-step" # keep it consistent with inference
ROLLOUT="single-rollout"     # keep it consistent with inference
WINDOW_SIZE=0                # keep it consistent with inference
SEGMENT_NUM=1                # keep it consistent with inference 
EVAL_MODE="single-scene"
WHICH_SCENE="final"

NUM_PROCESSES=1

ANNO="../../EXPLORE-Dataset/test_anno.json"

python eval.py \
    --data_root "$DATA_ROOT" \
    --anno "$ANNO" \
    --llm "$LLM" \
    --bert "$BERT" \
    --soft_coverage \
    --description_file "$DESCRIPTION_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --infer_strategy "$INFER_STRATEGY" \
    --eval_mode "$EVAL_MODE" \
    --window_size $WINDOW_SIZE \
    --segment_num $SEGMENT_NUM \
    --rollout "$ROLLOUT" \
    --which_scene "$WHICH_SCENE" \
    --num_processes $NUM_PROCESSES \
