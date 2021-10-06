#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
lr=5e-5

export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart-debug-0917

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=${GPUID}

python ${ROOT_DIR}/finetune_bart_amrparsing_new.py \
    --data_dir=${ROOT_DIR}/data/AMR17-parsing \
    --task amrparsing \
    --model_name_or_path ${MODEL} \
    --eval_batch_size=16 \
    --gpus 1 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length=512 \
    --max_target_length=1024 \
    --val_max_target_length=1024 \
    --test_max_target_length=1024 \
    --eval_max_gen_length=1024 \
    --do_predict \
    --seed 42 \
    --fp16 \
    --eval_beams ${eval_beam} 2>&1 | tee $OUTPUT_DIR/eval.log
