#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
lr=2e-5

export OUTPUT_DIR_NAME=outputs/Eval-AMR17-bart-base-amr2text-baseline
export TOKPATH=../../../data/pretrained-model/bart-base/
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}
export OMP_NUM_THREADS=10
export CUDA_VISIBLE_DEVICES=${GPUID}
mkdir -p $OUTPUT_DIR

python ${ROOT_DIR}/finetune_bart_amr2text.py \
    --data_dir=../data/AMR17-full \
    --task amrparsing \
    --model_name_or_path ${MODEL} \
    --tokenizer_name_or_path $TOKPATH \
    --eval_batch_size=8 \
    --gpus 1 \
    --max_epochs=1 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length=1024 \
    --max_target_length=384 \
    --val_max_target_length=384 \
    --test_max_target_length=384 \
    --eval_max_gen_length=384 \
    --do_predict \
    --seed 42 \
    --fp16 \
    --eval_beams ${eval_beam} 2>&1 | tee $OUTPUT_DIR/eval.log
