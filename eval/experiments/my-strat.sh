#!/bin/bash

DATA_ROOT="../../EXPLORE-Dataset"
LLM="../infer/Qwen/Qwen3-VL-2B-Instruct"          # path to llm scorer
BERT="./all-MiniLM-L6-v2"  # path to sbert

DESCRIPTION_FILE="../infer/sub_results/Qwen3-VL-2B-Instruct/my-strategy/Qwen3-VL-2B-Instruct.json"
OUTPUT_DIR="./subexp_scene_eval_res/Qwen3-VL-2B-Instruct"

INFER_STRATEGY="my-strategy" # keep it consistent with inference
EVAL_MODE="single-scene"
WHICH_SCENE="final"

NUM_PROCESSES=1

declare -a ANNOS=(
  "../../EXPLORE-Dataset/exp_anno_short.json"
  "../../EXPLORE-Dataset/exp_anno_med.json"
  "../../EXPLORE-Dataset/exp_anno_long.json"
  "../../EXPLORE-Dataset/exp_anno.json"
)

# declare -a ANNOS=(
#   "../../EXPLORE-Dataset/test_anno.json"
# )

declare -a DATASET_TYPES=(
 "short_seq"
 "medium_seq"
 "long_seq"
 "full"
)

# declare -a DATASET_TYPES=(
#  "short_seq"
# )

for i in "${!ANNOS[@]}"; do
  ANNO="${ANNOS[$i]}"
  DATASET_TYPE="${DATASET_TYPES[$i]}"
    
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
    --which_scene "$WHICH_SCENE" \
    --dataset_type "$DATASET_TYPE" \
    --num_processes $NUM_PROCESSES 
done
