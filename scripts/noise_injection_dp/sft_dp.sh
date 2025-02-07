#!/bin/bash
set -e

dataset_name="PHI_None-flashcard_10000-medmcqa_None-pubmedqa_1k_50000-val_size_0.1-max_input_length_1024"

n_epoch=4
delta=0.0001

epsilons=(1 2 5 10 100 1000)
for epsilon in ${epsilons[@]}; do
  init_model='meta-llama/Llama-3.2-3B'

  for ((epoch=1; epoch<=n_epoch; epoch++)); do
    output="Llama-3.2-3B-lora-dp-lr0.00001-eps-$epsilon-epoch-$epoch-PHI_duplicated-flashcards-medmcqa-pubmedqa"
    printf "Output name: $output\n"

    python3 $(dirname $0)/../../src/sft_dp.py \
      --model_name $init_model \
      --output $output \
      --dataset_name $dataset_name \
      --use_wandb True \
      --learning_rate 0.00001 \
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
      --optim 'sgd' \
      --max_grad_norm 0.0001 \
      --use_lora True \
      --lora_r 16 \
      --lora_alpha 8 \
      --lora_dropout 0.05 \
      --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]" \
      --eps $epsilon \
      --delta $delta

    init_model="/assets/models/dp/$output"
  done
done
