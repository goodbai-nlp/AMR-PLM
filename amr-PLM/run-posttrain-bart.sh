export CUDA_VISIBLE_DEVICES=0,1,2,3
dataset=AMR17-seq2seq
dataset=AMR17-new
datapath=data/$dataset
MODEL=$1
outpath=${dataset}-bart-large-debug-0925-1e-5
outpath=${dataset}-bart-large-mlmText-0928-1e-5
mkdir -p $outpath

python -m torch.distributed.launch --nproc_per_node=4 run_language_modeling_bart_amr.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/dev.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --add_tokens $datapath/add_tokens.json \
  --mlm \
  --block_size 256 \
  --per_gpu_train_batch_size 8 \
  --gradient_accumulation_steps 64  \
  --model_type "facebook/bart-large" \
  --model_name_or_path $MODEL \
  --save_total_limit 5 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate 0.00001 \
  --logging_steps 50 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
