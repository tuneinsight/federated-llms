#!/bin/bash
set -e

learning_rates=(0.000005 0.00001 0.00005 0.0001 0.0005)
dataset=("flashcard" "medmcqa" "pubmedqa")
for dataset in ${dataset[@]}
do
  for lr in ${learning_rates[@]}
  do
    dataset_name="federated/PHI_proportional_splits/${dataset}"
    output="Llama-3.1-8B-${dataset}/${lr}"
    printf "Dataset: $dataset_name\n"
    printf "Output name: $output\n"

    python ../../src/sft.py \
      --model_name 'meta-llama/Llama-3.2-3B' \
      --output $output \
      --dataset_name $dataset_name \
      --use_wandb True \
      --learning_rate $lr \
      --max_input_length 1024 \
      --num_epochs 1 \
      --max_steps -1 \
      --eval_steps 50 \
      --save_steps 50 \
      --gradient_checkpointing True \
      --global_batch_size 32 \
      --per_device_batch_size 16 \
      --flash_attention True \
      --train_in_8bit False \
      --optim 'adamw_torch' \
      --use_lora False \
      --lora_r 16 \
      --lora_alpha 8 \
      --lora_dropout 0.05 \
      --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]"
  done
done