#!/bin/bash
set -e

dataset_name="PHI_None-flashcard_10000-medmcqa_None-pubmedqa_1k_50000-val_size_0.1-max_input_length_1024"
noise_alphas=(60 100 200 500 1000)
for neftune_alpha in ${noise_alphas[@]}
do
  output="Llama-3.2-3B-lora-neftune-$neftune_alpha-PHI_duplicated-flashcards-medmcqa-pubmedqa"
  printf "Output name: $output\n"

  python3 $(dirname $0)/../../src/sft.py \
    --model_name 'meta-llama/Llama-3.2-3B' \
    --output $output \
    --dataset_name $dataset_name \
    --use_wandb True \
    --learning_rate 0.0005 \
    --max_input_length 1024 \
    --num_epochs 4 \
    --max_steps -1 \
    --eval_steps 150 \
    --save_steps 150 \
    --gradient_checkpointing False \
    --global_batch_size 32 \
    --per_device_batch_size 8 \
    --flash_attention True \
    --train_in_8bit False \
    --optim 'paged_adamw_32bit' \
    --neftune_noise_alpha $neftune_alpha \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]"
done
