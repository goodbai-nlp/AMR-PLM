export CUDA_VISIBLE_DEVICES=1,5,6,7
dataset=AMR20-merged
datapath=data/$dataset
MODEL=$1
lr=1e-5
outpath=${dataset}-bart-base-textinf-2task-${lr}
mkdir -p $outpath

python -u -m torch.distributed.launch --nproc_per_node=4 run_textinfilling_bart_amr.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/val.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 128  \
  --model_type "facebook/bart-base" \
  --model_name_or_path $MODEL \
  --save_total_limit 3 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --logging_steps 50 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
