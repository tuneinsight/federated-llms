#!/bin/bash
set -e

# Get the fine tuning and aggregation scripts
agg_script_name="aggregate.sh"
agg_script_dir="$(dirname "$0")"
agg_script_path="$agg_script_dir/$agg_script_name"

# Set the models to aggregate
assets='/assets'

##################################################################################################
#### Federated learning Params ###################################################################
##################################################################################################
init_round=0

# To recover from suspended job
# base_model="/assets/merged_models/Llama-3.2-3B_lora_OFF_2024_11_22_10_19/round_1/"
# init_round=2

base_model="meta-llama/Llama-3.1-8B"
model_name="Llama-3.1-8B"

datasets=("flashcard" "medmcqa" "pubmedqa")

use_lora=False
n_rounds=5
timestamp=2024_11_25_13_19
##################################################################################################


if [[ "$use_lora" == "True" ]]; then
    lora_status="lora_ON"
    gradient_checkpointing=False
    per_device_batch_size=4
    optim=paged_adamw_32bit

    # Set lr according to model
    case "$model_name" in
    "Llama-2-7b-hf")
        learning_rates=(0.002 0.001 0.0005)
        ;;
    "Llama-3.1-8B")
        learning_rates=(0.001 0.0005 0.0005)
        ;;
    "Llama-3.2-3B")
        learning_rates=(0.001 0.001 0.001)
        ;;
    *)
        echo "No LR setting for model name: $model_name"
        exit 1
        ;;
    esac
else
    lora_status="lora_OFF"
    gradient_checkpointing=True
    per_device_batch_size=16
    optim=adamw_torch

    # Set lr according to model
    case "$model_name" in
    "Llama-2-7b-hf")
        learning_rates=(0.0001 0.0001 0.00001)
        ;;
    "Llama-3.1-8B")
        learning_rates=(0.0001 0.0001 0.00001)
        ;;
    "Llama-3.2-3B")
        learning_rates=(0.00005 0.0001 0.00002)
        ;;
    "Mistral-7B-v0.3")
        learning_rates=(0.00001 0.00001 0.00001)
        ;;
    *)
        echo "No LR setting for model name: $model_name"
        exit 1
        ;;
    esac
fi

n_participants=${#datasets[@]}

init_model=$base_model

for ((round = init_round; round < n_rounds; round++)); do
    output_models=()
    for ((i = 0; i < n_participants; i++)); do


        # To recover from suspended job in a round
        if [[ "$round" == 0 && "$i" != 0 ]]; then
            printf "Skipping dataset $dataset_name"
            continue
        fi

        dataset_name=${datasets[$i]}
        dataset_path="federated/PHI_proportional_splits/$dataset_name/"
        learning_rate=${learning_rates[$i]}
        output_model="$assets/models/federated/${model_name}_${lora_status}_${timestamp}/round_${round}/$dataset_name"
        output_models+=($output_model)

        printf "\nFederated learning round $round, init model: $init_model, lora: $use_lora, dataset: $dataset_name, lr=$learning_rate\n"

        python $(dirname $0)/../src/sft.py \
            --model_name "$init_model" \
            --output "$output_model" \
            --dataset_name "$dataset_path" \
            --use_wandb True \
            --learning_rate $learning_rate \
            --max_input_length 1024 \
            --num_epochs 1 \
            --max_steps -1 \
            --eval_steps 50 \
            --save_steps 50 \
            --gradient_checkpointing $gradient_checkpointing \
            --global_batch_size 32 \
            --per_device_batch_size $per_device_batch_size \
            --flash_attention True \
            --train_in_8bit False \
            --optim "$optim" \
            --use_lora $use_lora \
            --lora_r 16 \
            --lora_alpha 8 \
            --lora_dropout 0.05 \
            --lora_target_modules "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"o_proj\",\"down_proj\",\"gate_proj\"]"
    done

    merged_output="$assets/merged_models/${model_name}_${lora_status}_${timestamp}/round_$round"
    mkdir -p "$merged_output"
    IFS=','
    models="${output_models[*]}"

    printf "\nAggregating round $round...\n"
    source "$agg_script_path"
    init_model=$merged_model_path

    # Copy tokenizer of the source models
    first_model=${output_models[0]}
    cp ${first_model}/tokenizer_config.json $merged_model_path
    # Copy tokenizer.model if it exists
    tokenizer_file=${first_model}/tokenizer.model
    [[ -e $tokenizer_file ]] && cp $tokenizer_file $merged_model_path
    # Copy tokenizer.json if it exists
    tokenizer_file=${first_model}/tokenizer.json
    [[ -e $tokenizer_file ]] && cp $tokenizer_file $merged_model_path

    printf "\nAggregation done, model stored in: $merged_model_path"
done
