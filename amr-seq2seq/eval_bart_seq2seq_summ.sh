#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

DATA_BASE=$3
CHECK_POINT=$2
MODEL=$1
GPUID=0

export OUTPUT_DIR_NAME=outputs/test-bart-large-local-summ-0717
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

echo "DataBase:$DATA_BASE"
echo "CheckPoint:$CHECK_POINT"

rm -rf $OUTPUT_DIR

#mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10


export CUDA_VISIBLE_DEVICES=${GPUID}
python ${ROOT_DIR}/finetune_bart_seq2seq.py \
--data_dir=${DATA_BASE} \
--task summarization \
--model_name_or_path=${MODEL} \
--eval_batch_size=4 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--checkpoint=$CHECK_POINT \
--max_source_length=512 \
--max_target_length=100 \
--val_max_target_length=100 \
--test_max_target_length=100 \
--eval_max_gen_length=100 \
--do_predict \
--seed 42 \
--eval_beams 5
