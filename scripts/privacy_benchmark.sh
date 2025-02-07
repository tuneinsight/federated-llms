#!/bin/bash
set -e

assets='/assets'

if [ -z $load_generations ]; then
    load_generations=True
fi
echo "Load generations is set to $load_generations"

if [ -z $checkpoint_path ]; then
    ## Local
    model_name="Llama-3.2-3B-PHI_duplicated-flashcards-medmcqa-pubmedqa"
    split="0.00005"
    checkpoint_path="$assets/models/$model_name/$split"
fi
echo "Running privacy benchmark on <$checkpoint_path>..."

training_dataset="PHI_None-flashcard_10000-medmcqa_None-pubmedqa_1k_50000-val_size_0.1-max_input_length_1024"
if [ -z $training_dataset_path ]; then 
    training_dataset_path="$assets/datasets/$training_dataset"
fi
echo "Training dataset for privacy generation: <$training_dataset_path>..."

if [ -z $privacy_output ]; then 
    privacy_output="$assets/results/privacy-benchmark/$model_name/$split"
fi
echo "Privacy benchmarks output: <$privacy_output>..."

PHI_dataset="$assets/datasets/PHI_datasets/PHI_dataset_duplicated_0.3_seed_42.json"

# args="--checkpoint_path $checkpoint_path --output_dir $output_dir --load_generations $load_generations --"
python $(dirname $0)/../src/privacy_benchmark.py \
  --model_checkpoint $checkpoint_path \
  --output_dir $privacy_output \
  --load_generations $load_generations \
  --training_dataset $training_dataset_path \
  --PHI_dataset $PHI_dataset
