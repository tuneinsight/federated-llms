#!/bin/bash
set -e

assets="/assets"

if [ -z $checkpoint_path ]; then
    model_name="Llama3-8B-PHI_duplicated-flashcards-medmcqa-pubmedqa"
    lr="0.00005"

    checkpoint_path="$assets/models/$model_name/$lr/"
fi
echo "Running utility benchmark on <$checkpoint_path>..."

run_name=$(echo "$checkpoint_path" | awk -F'/' '{print $(NF-2)}')
shots=3

out_dir="$assets/results/utility-benchmark/generations"

inference_args="--checkpoint $checkpoint_path --checkpoint_name $run_name --shots $shots --batch_size 32" #  --multi_seed 
evaluate_args="--checkpoint $run_name --shots $shots --out_dir $out_dir"
benchmarks=("mmlu_medical" "pubmedqa" "medmcqa" "medqa" "medqa4")

benchmarks=("mmlu_medical" "pubmedqa" "medmcqa" "medqa" "medqa4")
for i in ${benchmarks[@]}
do
    printf "\n================= Inference on $i - $model_name - $shots shots ===================\n"
    printf "\npython inference.py $inference_args --benchmark $i\n\n"
    python $(dirname $0)/../src/evaluation/inference.py $inference_args --benchmark $i
    python $(dirname $0)/../src/evaluation/evaluate.py $evaluate_args --benchmark $i
done


printf "\n\n================= Evaluating $checkpoint ===================\n\n"
# printf "\n"
for i in ${benchmarks[@]}
do
    python $(dirname $0)/../src/evaluation/evaluate.py $evaluate_args --benchmark $i
done
