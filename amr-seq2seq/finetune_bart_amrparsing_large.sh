#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
lr=3e-5

export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-0917
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-0924
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-0925
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-0925-debug
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-TAPT-Text
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-TAPT-Text-checkpoint450
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-amrtoken
export OUTPUT_DIR_NAME=outputs/AMR17-bart-large-amrparsing-baseline-lr${lr}-init-noearly-beam${eval_beam}-leonard-smart2-debug-amrtoken-py38torch1.8

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10

export CUDA_VISIBLE_DEVICES=${GPUID}
python -u ${ROOT_DIR}/finetune_bart_amrparsing_new.py \
    --data_dir=${ROOT_DIR}/data/AMR17-parsing \
    --learning_rate=$lr \
    --num_train_epochs 20 \
    --task amrparsing \
    --model_name_or_path=${MODEL} \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --accumulate_grad_batches 2 \
    --early_stopping_patience 15 \
    --gpus 1 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length=512 \
    --max_target_length=1024 \
    --val_max_target_length=1024 \
    --test_max_target_length=1024 \
    --eval_max_gen_length=1024 \
    --do_train --do_predict \
    --seed 42 \
    --smart_init \
    --eval_beams ${eval_beam} 2>&1 | tee $OUTPUT_DIR/run.log
