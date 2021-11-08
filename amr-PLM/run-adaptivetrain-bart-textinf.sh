export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,3
dataset=Taskdata

datapath=../data/$dataset
tokpath=../../data/pretrained-model/bart-base
MODEL=$1
interval=1

lr=1e-5

outpath=output/${dataset}-bart-base-textinf-${lr}-TaskAdaptivePLM
outpath=output/${dataset}-bart-base-textinf-${lr}-TaskAdaptivePLM-AMRbart4task

mkdir -p $outpath

#python -u -m torch.distributed.launch --nproc_per_node=4 run_textinfilling_taskAdaptive.py \
python -u -m torch.distributed.launch --nproc_per_node=2 --master_port 86886 run_textinfilling_taskAdaptive.py \
  --train_file $datapath/train.txt \
  --val_file $datapath/val.txt \
  --output_dir $outpath \
  --mlm_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 1  \
  --model_type "facebook/bart-base" \
  --model_name_or_path $MODEL \
  --tokenizer_name_or_path $tokpath \
  --save_total_limit 3 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 1000 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
