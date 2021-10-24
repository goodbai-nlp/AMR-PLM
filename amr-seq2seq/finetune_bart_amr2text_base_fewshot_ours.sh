#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
num_ins=128
lr=1e-5
lr=2e-5
# lr=5e-5

export OUTPUT_DIR_NAME=outputs/AMR17-bart-base-ours-amr2text-fewshot-$num_ins-$lr
Tokenizer=../../../data/pretrained-model/bart-base

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}
python ${ROOT_DIR}/finetune_bart_amr2text.py \
    --data_dir=data/AMR17-${num_ins}ins \
    --learning_rate=$lr \
    --num_train_epochs 60 \
    --task graph2text \
    --model_name_or_path=${MODEL} \
    --tokenizer_name_or_path=${Tokenizer} \
    --train_batch_size=8 \
    --eval_batch_size=4 \
    --accumulate_grad_batches 1 \
    --early_stopping_patience 15 \
    --gpus 1 \
    --output_dir=$OUTPUT_DIR \
    --max_source_length=1024 \
    --max_target_length=384 \
    --val_max_target_length=384 \
    --test_max_target_length=384 \
    --eval_max_gen_length=384 \
    --do_train --do_predict \
    --seed 42 \
    --fp16 \
    --eval_beams ${eval_beam} 2>&1 | tee $OUTPUT_DIR/run.log

