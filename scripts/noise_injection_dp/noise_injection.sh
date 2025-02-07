#!/bin/bash
set -e

init_model="/assets/models/Llama-3.2-3B-lora-PHI_duplicated-flashcards-medmcqa-pubmedqa/0.0005/"

sigmas=(0 0.001 0.01 0.02 0.03 0.04 0.05 0.06)
for sigma in ${sigmas[@]}; do
  output="Llama-3.2-3B-lora-noise-sigma-$sigma"
  printf "Output name: $output\n"

  python3 $(dirname $0)/../../src/noise.py \
    --model_name $init_model \
    --output $output \
    --dataset_name empty \
    --use_wandb True \
    --learning_rate 0.0005 \
    --max_input_length 1024 \
    --num_epochs 1 \
    --max_steps -1 \
    --eval_steps 150 \
    --save_steps 150 \
    --gradient_checkpointing False \
    --global_batch_size 32 \
    --per_device_batch_size 8 \
    --flash_attention True \
    --train_in_8bit False \
    --optim 'paged_adamw_32bit' \
    --max_grad_norm 0.0001 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]" \
    --sigma $sigma
done
