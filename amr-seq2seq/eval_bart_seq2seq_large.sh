#!/bin/bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MODEL=$1
GPUID=$2

export OUTPUT_DIR_NAME=outputs/test-bart-large-baseline-fp16
export OUTPUT_DIR_NAME=outputs/test-bart-large-baseline-fp16-debug
export DATA_BASE=${ROOT_DIR}/data/AMR17
export CURRENT_DIR=${ROOT_DIR}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

echo "DataBase:$DATA_BASE"

rm -rf $OUTPUT_DIR

mkdir -p $OUTPUT_DIR

export OMP_NUM_THREADS=10

export CUDA_VISIBLE_DEVICES=${GPUID}
python ${ROOT_DIR}/finetune_bart_seq2seq.py \
--data_dir=${DATA_BASE} \
--task graph2text \
--model_name_or_path=${MODEL} \
--eval_batch_size=16 \
--gpus 1 \
--output_dir=$OUTPUT_DIR \
--checkpoint='./' \
--max_source_length=1024 \
--max_target_length=384 \
--val_max_target_length=384 \
--test_max_target_length=384 \
--eval_max_gen_length=384 \
--do_predict \
--seed 42 \
--fp16 \
--eval_beams 5 | tee ${OUTPUT_DIR}/eval.log
