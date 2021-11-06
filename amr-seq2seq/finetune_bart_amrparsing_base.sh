#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

GPUID=$2
MODEL=$1
eval_beam=5
lr=5e-5

export OUTPUT_DIR_NAME=outputs/AMR17-bart-base-amrparsing-lr${lr}-baseline

export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

if [ ! -d $OUTPUT_DIR ];then
  mkdir -p $OUTPUT_DIR
else
  echo "${OUTPUT_DIR} already exists, change a new one or delete origin one"
  exit 0
fi

export OMP_NUM_THREADS=10

export CUDA_VISIBLE_DEVICES=${GPUID}
python -u ${ROOT_DIR}/finetune_bart_amrparsing.py \
    --data_dir=../data/AMR17-full \
    --learning_rate=$lr \
    --num_train_epochs 20 \
    --task amrparsing \
    --model_name_or_path=${MODEL} \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --accumulate_grad_batches 1 \
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
    --fp16 \
    --eval_beams ${eval_beam} 2>&1 | tee $OUTPUT_DIR/run.log
