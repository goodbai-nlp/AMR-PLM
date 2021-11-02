export CUDA_VISIBLE_DEVICES=0,1,2,3
dataset=AMR20-full
datapath=../data/$dataset
MODEL=$1
lr=3e-5
interval=1
outpath=output/${dataset}-bart-base-textinf-JointDenoise-2task-${lr}

mkdir -p $outpath

python -u -m torch.distributed.launch --nproc_per_node=4 run_textinfilling_bart_denoising_4task.py \
  --train_file $datapath/train.jsonl \
  --val_file $datapath/val.jsonl \
  --test_file $datapath/test.jsonl \
  --output_dir $outpath \
  --mlm \
  --mlm_amr \
  --mlm_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 1  \
  --model_type "facebook/bart-base" \
  --model_name_or_path $MODEL \
  --save_total_limit 3 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 60000 \
  --logging_steps 1000 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log