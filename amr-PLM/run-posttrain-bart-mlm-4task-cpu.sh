dataset=AMR17-new
datapath=data/$dataset
MODEL=$1
lr=0.00005
outpath=${dataset}-bart-base-mlm4task-${lr}-cpu
mkdir -p $outpath

python run_language_modeling_bart_amr.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/dev.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --mlm_amr_plus_text \
  --mlm_text_plus_amr \
  --block_size 512 \
  --per_gpu_train_batch_size 2 \
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
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
