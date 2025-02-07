#!/bin/bash

CUDA_VISIBLE_DEVICES=1

output_dir="$(pwd)/output/merged"

assets='/assets'

if [ -z "$models" ]; then 
    model_split_0="$assets/models/federated/Llama-2-7b-hf_lora_OFF/round_0/flashcard"
    model_split_1="$assets/models/federated/Llama-2-7b-hf_lora_OFF/round_0/medmcqa"
    model_split_2="$assets/models/federated/Llama-2-7b-hf_lora_OFF/round_0/pubmedqa"

    models="$model_split_0,$model_split_1,$model_split_2"
fi

# merged_output="$assets/merged_models/test"
# if merged_output is set, it overrides the output_dir setting
if [ -n $merged_output ]; then
    export MERGED_OUTPUT=$merged_output
    output_dir=$merged_output
fi

cd $(dirname $0)/../OpenFedLLM

num_clients=$(echo "$models" | awk -F',' '{print NF}')

agg_output=$(python3 main_sft_aggregation.py \
 --model_name_or_path "$models" \
 --fed_alg "fedavg" \
 --sample_clients $num_clients \
 --max_steps 1 \
 --num_rounds 1 \
 --batch_size 2 \
 --gradient_accumulation_steps 1 \
 --seq_length 512 \
 --peft_lora_r 32 \
 --peft_lora_alpha 64 \
 --use_peft \
 --output_dir "$output_dir")

cd -

last_line=$(echo "$agg_output" | tail -n 1)
merged_model_path="$last_line"
