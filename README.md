# FederatedLLMs
This repository contains the code used to achieve the results presented in our paper: "Mitigating Unintended Memorization with LoRA in Federated Learning for LLMs". It contains scripts and tools to fine-tune, evaluate, and operate on open Large Language Models (LLMs) using centralized and federated learning techniques, as well as notebooks to create the datasets used in our experiments.

## Repository Structure
* `./src/`: Contains Python scripts for various LLM operations.
* `./scripts/`: Includes Bash scripts that facilitate fine-tuning and evaluation processes in different settings. More details are provided in the Scripts section.

## Reference libraries
This repository integrates code from the following external projects:
1. [Meditron](https://github.com/epfLLM/meditron): The `./src/evaluation` folder includes code from Meditron to create benchmark evaluations.
2. [OpenFedLLM](https://github.com/Feng-Hong/OpenFedLLM): The `./OpenFedLLM` folder includes code from OpenFedLLM, used to merge LLMs and LoRAs in the federated learning scenarios.

## Installation
To set up the environment and install dependencies, follow these steps:
```
conda create -n fedllms python=3.11
conda activate fedllms
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r ./pip_requirements.txt
```

## Data
### Fine-tuning datasets
Our fine-tuning datasets include *PubMedQA*, *MedMCQA* and *Medical Meadow Flashcards*, from which canaries for privacy evaluation have been injected with a rate proportional to the size of the datasets. See [Sources](#sources) for more details about the datasets. In the centralized setting, the fine-tuning dataset is an aggregate of the 3 datasets containing injected canaries. In our 3-participants' federated learning experiments, each virtual contributor use one of the 3 datasets with canaries injected during the local fine-tuning steps that precede aggregations.

#### Sources
In our centralized and federated experiments, the 3 following popular medical datasets with different types of QA were used:

1. **MedMCQA** is composed of multiple-choice questions, containing almost 190k entrance exam questions (AIIMS & NEET PG).
source: [(Pal et al., 2022)](https://proceedings.mlr.press/v174/pal22a.html)

2. **PubMedQA** consists of Yes/No/Maybe questions created from PubMed abstracts. The dataset contains 1k expert-annotated (PQA-L) and 211k artificially generated QA instances (PQA-A). We include 500 questions from the train and validation sets of PQA-L and 50k questions of PQA-A. 
source: [(Jin et al., 2019)](https://aclanthology.org/D19-1259/)

3. **Medical Meadow flashcards** contains 39k questions created from Anki Medical Curriculum flashcards compiled by medical students. We include 10k samples for fine-tuning data.
source: https://arxiv.org/abs/2304.08247

4. **i2b2 UTHealth** contains 1304 longitudinal medical records that describe 296 patients.
source: https://www.sciencedirect.com/science/article/pii/S1532046415001823

### Creating the fine-tuning datasets

1. Request access and download the i2b2 dataset from the [n2c2 NLP Research Data Sets](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/). The i2b2 dataset has been released in the track _2014 - Deidentification & Heart Disease_ under _Annotating longitudinal clinical narratives for de-identification: The 2014 i2b2/UTHealth corpus_.
2. We preprocess the records to remove blank spaces and homogenize records, using the notebook `notebooks/PHI_dataset_processing.ipynb`, which results in a `PHI_dataset.json` file
3. Create the folder `/assets` where checkpoints, datasets and benchmark results will be saved. Across our repository this folder is refered as `assets` or `ASSETS`. You can modify the name and path of the `assets` folder but make sure to change it consistently across the repository files relying on this folder.
4. We create the centralized fine-tuning dataset using `scr/create_dataset.py`. This is were we inject the medical records and choose the ratio of duplicated documents. The script creates two datasets: 1) the fine-tuning dataset, which names depend of the settings you choose, e.g. `assets/dataset/PHI_None-flashcard_10000-medmcqa_None-pubmedqa_1k_50000-val_size_0.1-max_input_length_1024` and 2) a dataset containing the duplicated medical records `assets/datasets/PHI_datasets/PHI_dataset_duplicated_0.3_seed_42.json.json`.
5. To create the 3 federated datasets, we rely on `src/create_dataset.py` to create 4 individual datasets: `flashcard`, `pubmedqa`, `medmcqa` and `PHI`. We then use `notebooks/federated_dataset.ipynb` to split `PHI` and inject it into the other 3 proportionally.

## Scripts
This repository provides various scripts for fine-tuning, evaluating utility and privacy, injecting noise, and running federated learning experiments. They are listed below:

### Fine-tuning
* `./scripts/sft.sh` fine-tunes an LLM using either a local path or a HuggingFace reference. See Transformers' [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) and [TrainingArguments](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/trainer#transformers.TrainingArguments) for parameter details.

### Evaluating Utility & Privacy
* `./scripts/utility_benchmark.sh`: Evaluates the given model's utility through *mmlu_medical*, *pubmedqa*, *medmcqa*, *medqa* and *medqa4* benchmarks.
* `./scripts/privacy_benchmark.sh`: Assesses privacy by measuring memorization scores on LLMs fine-tuned with synthetic injection of private sentences.

### Differential Privacy (DP) and noise injection
* `./scripts/noise_injection_dp/sft_dp.sh` fine-tunes an LLM, similarly to `./scripts/sft.sh`, with (ϵ,δ)-DP noise. Note: this requires `sgd` optimizer.
* `./scripts/noise_injection_dp/noise_injection.sh` injects random gaussian noise of variable standard deviation (sigma) to the weights of fine-tuned models.

### Federated Learning (FL)
* `./scripts/federated/federated_learning.sh`: Simulates collective fine-tuning using FedAvg with 3 virtual participants utilizing *flashcard* *medmcqa* and *pubmedqa* datasets respectively.

### Hyperparameter search
The `./scripts/sweeps` folder contains scripts for sequentially fine-tuning a base model in a centralized setting with varying parameters. The hyperparameters search scripts include:
* `./scripts/sweeps/sft_search_batch_size` for selecting the batch size used during fine-tuning.
* `./scripts/sweeps/sft_search_gradclip` for selecting an optimal gradient clipping required for fine-tuning LLMs with differential privacy.
* `./scripts/sweeps/sft_search_lr_datasets` for selecting the learning rate of the optimizer for each of the local datasets used in the federated learning experiments.
* `./scripts/sweeps/sft_search_lr` for selecting the learning rate of the optimizer used in centralized fine-tuning.
* `./scripts/sweeps/sft_search_neftune` for gradient clipping selection
