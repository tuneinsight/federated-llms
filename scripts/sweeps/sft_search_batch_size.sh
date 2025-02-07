#!/bin/bash
set -e

dataset_name="PHI_None-flashcard_10000-medmcqa_None-pubmedqa_1k_50000-val_size_0.1-max_input_length_1024"
output="Llama-3.2-3B-bs-256-lr-0.00014-PHI_duplicated-flashcards-medmcqa-pubmedqa"
printf "Output name: $output\n"

python ../../src/sft.py \
    --model_name 'meta-llama/Llama-3.2-3B' \
    --output $output \
    --dataset_name $dataset_name \
    --use_wandb True \
    --learning_rate 0.00014 \
    --max_input_length 1024 \
    --num_epochs 4 \
    --max_steps -1 \
    --eval_steps 18 \
    --save_steps 18 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --global_batch_size 256 \
    --per_device_batch_size 16 \
    --flash_attention True \
    --train_in_8bit False \
    --optim 'adamw_torch' \
    --use_lora False \
    --lora_r 16 \
    --lora_alpha 8 \
    --lora_dropout 0.05 \
    --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]"