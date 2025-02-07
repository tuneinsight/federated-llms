import fire
import json
import random
from os.path import join
from typing import Union
from datetime import datetime
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import AutoTokenizer

RESPONSE_TEMPLATE = "The answer is:"
DATASET_FOLDER = '/assets/datasets/'
PHI_DATASETS = '../../assets/datasets/PHI_datasets/'

def duplicate_PHI_documents(seed=42, n_samples=None, duplicated_ratio=0.3):
    with open(join(PHI_DATASETS, 'PHI_dataset.json'), 'r') as f:
        dataset = json.load(f)
    random.seed(seed)
    random.shuffle(dataset)

    if n_samples:
        if len(dataset) < n_samples:
            print(f"n_samples > PHI dataset size ({n_samples} > {len(dataset)}). Using the whole dataset.")
        dataset = dataset[:n_samples]
        assert len(dataset) <= n_samples

    split_idx = int(len(dataset) * duplicated_ratio)
    duplicated_10x = dataset[:split_idx]
    non_duplicated = dataset[split_idx:]
    assert len(dataset) == len(non_duplicated) + len(duplicated_10x)
    print("Non duplicated split size:", len(non_duplicated))
    print("Duplicated 10x split size:", len(duplicated_10x))
    print("Duplicated dataset size:", len(non_duplicated) + 10*len(duplicated_10x))

    for note in non_duplicated:
        note['duplication'] = 1
    for note in duplicated_10x:
        note['duplication'] = 10
    dataset = non_duplicated + duplicated_10x

    filename = f"PHI_dataset_duplicated_{duplicated_ratio}_seed_{seed}"
    if n_samples:
        filename  += f"_n_samples_{n_samples}"
    filename += ".json"
    filepath = join(PHI_DATASETS, filename) 
    print(f"Saving duplicated dataset at {filepath}")
    with open(filepath, 'w') as f:
        f.write(json.dumps(dataset, indent=2))
    return filepath

def create_PHI_injection_dataset(tokenizer, max_input_length, filepath=None, train_test_split=False, val_set_size=0.1):
    if filepath:
        phi_dataset = Dataset.from_json(filepath)
    else:
        # Non duplicated dataset
        phi_dataset = Dataset.from_json(join(PHI_DATASETS, 'PHI_dataset.json'))

    # Filter question + answer smaller than max_input_length
    def filter_and_duplicate_notes(examples):
        texts = []
        for example_text, example_duplication in zip(examples['text'], examples['duplication']):
            text = tokenizer.decode(tokenizer(example_text)['input_ids'][:max_input_length])
            text = f"""Medical Note: {example_text}\n"""
            texts += [text] * example_duplication
        return {'text': texts}
    phi_dataset = phi_dataset.select_columns(['text', 'duplication']).map(filter_and_duplicate_notes, batched=True, batch_size=32, remove_columns=['duplication'])

    # Split huggingface train split into train-validation
    if train_test_split:
        phi_dataset = phi_dataset.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        phi_dataset["validation"] = phi_dataset.pop("test") # rename test into validation

    phi_dataset = phi_dataset.select_columns(['text'])
    return phi_dataset

def create_flashcard_dataset(tokenizer, val_set_size, max_input_length, n_samples=None):
    flashcard_name = 'medalpaca/medical_meadow_medical_flashcards'
    split = 'train'
    if n_samples:
        split += f"[:{n_samples}]"
    flashcard_dataset = load_dataset(flashcard_name, split=split)

    # Preprocess the dataset
    initial_size = len(flashcard_dataset)
    # Filter question + answer smaller than max_input_length
    flashcard_dataset = flashcard_dataset.select_columns(['input', 'output'])\
        .rename_columns({"input": "question", 'output': 'answer'})\
        .filter(lambda row: len(tokenizer(row['question'] + row['answer'])['input_ids']) <= max_input_length)
    print("Flashcard preprocessing removed", initial_size - len(flashcard_dataset))

    # Split huggingface train split into train-validation
    flashcard_dataset = flashcard_dataset.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    flashcard_dataset["validation"] = flashcard_dataset.pop("test") # rename test into validation
    # Format the questions
    def prompt_format_flashcard(example):
        text = f"""Open Question: {example['question']}\n{RESPONSE_TEMPLATE} {example['answer']}\n"""
        return {'text': text}
    
    flashcard_dataset = flashcard_dataset.map(prompt_format_flashcard).select_columns(['text'])

    return flashcard_dataset

def create_medmcqa_dataset(tokenizer, val_set_size, max_input_length, n_samples=None):
    medmcqa_name = 'openlifescienceai/medmcqa'
    split = 'train'
    if n_samples:
        split += f"[:{n_samples}]"
    medmcqa_dataset = load_dataset(medmcqa_name, split=split)

    # Preprocess the dataset
    initial_size = len(medmcqa_dataset)
    # Filter input smaller than max_input_length 
    medmcqa_dataset = medmcqa_dataset.select_columns(['question', 'cop', 'opa', 'opb', 'opc', 'opd'])\
        .filter(lambda row: len(tokenizer(row['question'])['input_ids']) <= max_input_length )\
        .rename_columns({"cop": "answer", 'opa': 'option_a', 'opb': 'option_b', 'opc': 'option_c', 'opd': 'option_d'})\
        .filter(lambda row: row['answer'] in [0,1,2,3])
    print("MedMCQA preprocessing removed", initial_size - len(medmcqa_dataset))
    
    # Split huggingface train split into train-validation
    medmcqa_dataset = medmcqa_dataset.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    medmcqa_dataset["validation"] = medmcqa_dataset.pop("test") # rename test into validation

    # Format the dataset
    def format_promp_medmcqa(example):
        # Format question
        question = example['question']
        if not question.endswith('?') and not question.endswith('.') and not question.endswith(':') and not question.endswith('-'):
            question += '?'
        # Format answer
        answer_number = example['answer']
        answer_letter = chr(ord('A') + answer_number)
        example['option_a'] = 'A. ' + example['option_a']
        example['option_b'] = 'B. ' + example['option_b']
        example['option_c'] = 'C. ' + example['option_c']
        example['option_d'] = 'D. ' + example['option_d']
        # Put everything together
        text = f"""MCQ: {question}\nOptions:\n{example['option_a']}\n{example['option_b']}\n{example['option_c']}\n{example['option_d']}\n{RESPONSE_TEMPLATE} {answer_letter}\n""" #
        return {'text': text}
    
    medmcqa_dataset = medmcqa_dataset.map(format_promp_medmcqa).select_columns(['text'])
    return medmcqa_dataset

def create_pubmedqa_dataset(tokenizer, val_set_size, max_input_length, n_artificial_samples=None):
    pubmedqa_name = 'bigbio/pubmed_qa'

    artificial_split = 'train'
    if n_artificial_samples:
        artificial_split += f"[:{n_artificial_samples}]"
    pubmedqa_dataset = concatenate_datasets([
        load_dataset(pubmedqa_name, "pubmed_qa_labeled_fold0_source", split="train+validation"),
        load_dataset(pubmedqa_name, "pubmed_qa_artificial_source", split=artificial_split)
    ])
    # Preprocess
    initial_size = len(pubmedqa_dataset)
    pubmedqa_dataset = pubmedqa_dataset.select_columns(['QUESTION', 'CONTEXTS', 'final_decision'])\
        .rename_columns({"CONTEXTS": "context", "QUESTION": "question", 'final_decision': 'answer'})\
        .filter(lambda row: len(tokenizer(row['question'] + '\n'.join(row['context']))['input_ids']) <= max_input_length )\
        .filter(lambda row: row['answer'] in ['yes', 'no', 'maybe'])
    print("PubMedQA preprocessing removed", initial_size - len(pubmedqa_dataset))
    # Train val split
    pubmedqa_dataset = pubmedqa_dataset.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    pubmedqa_dataset["validation"] = pubmedqa_dataset.pop("test") # rename test into validation
    
    # Format questions
    def format_promp_pubmedqa(example):
        # Format question
        question = example['question']
        if not question.endswith('?') and not question.endswith('.') and not question.endswith(':') and not question.endswith('-'):
            question += '?'
        context = '\n'.join(example['context'])
        # Put everything together
        text = f"""Closed Question: {context}\n{question}\n{RESPONSE_TEMPLATE} {example['answer']}\n"""
        return {'text': text}
    
    pubmedqa_dataset = pubmedqa_dataset.map(format_promp_pubmedqa).select_columns(['text'])
    return pubmedqa_dataset

def main(output_name: str = None, val_set_size: Union[int, float] = 0.1, max_input_length: int = 1024):
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False)

    datasets = []
    dataset_names = []
    dataset_filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    n_samples = None
    PHI_filepath = duplicate_PHI_documents(seed=42, n_samples=n_samples)
    phi_dataset = create_PHI_injection_dataset(tokenizer, max_input_length, PHI_filepath, train_test_split=True, val_set_size=val_set_size)
    dataset_filename += f"-PHI_{n_samples}"
    print("PHI injection dataset", phi_dataset)
    print("PHI injection sample:\n")
    print("PHI dataset size:", phi_dataset['train'].data.nbytes / 1e6)
    print(phi_dataset['train'][0]['text'][:200], '\n')
    datasets.append(phi_dataset)
    dataset_names.append('phi')
    
    n_samples = 10000
    flashcard_dataset = create_flashcard_dataset(tokenizer, val_set_size, max_input_length, n_samples=n_samples)
    dataset_filename += f"-flashcard_{n_samples}"
    print("Flashcard dataset", flashcard_dataset)
    print("Flashcard dataset sample:\n")
    print(flashcard_dataset['train'][0]['text'], '\n')
    print("Flashcard dataset size:", flashcard_dataset['train'].data.nbytes / 1e6)
    datasets.append(flashcard_dataset)
    dataset_names.append('flashcard')

    n_samples = None # all the dataset
    medmcqa_dataset = create_medmcqa_dataset(tokenizer, val_set_size, max_input_length, n_samples=n_samples)
    dataset_filename += f"-medmcqa_{n_samples}"
    print("Medmcqa dataset", medmcqa_dataset)
    print("Medmcqa dataset sample:\n")
    print("MedMCQA dataset size:", medmcqa_dataset['train'].data.nbytes / 1e6)
    print(medmcqa_dataset['train'][0]['text'], '\n')
    datasets.append(medmcqa_dataset)
    dataset_names.append('medmcqa')

    n_samples = 50000
    pubmedqa_dataset = create_pubmedqa_dataset(tokenizer, val_set_size, max_input_length, n_artificial_samples=n_samples)
    dataset_filename += f"-pubmedqa_1k_{n_samples}"
    print("PubMedQA dataset", pubmedqa_dataset)
    print("PubMedQA dataset sample:\n")
    print("PubMedQA dataset size:", pubmedqa_dataset['train'].data.nbytes / 1e6)
    print(pubmedqa_dataset['train'][0]['text'], '\n')
    datasets.append(pubmedqa_dataset)
    dataset_names.append('pubmedqa')


    for dataset, name in zip(datasets, dataset_names):
        dataset = DatasetDict({
            "train": dataset['train'],
            "validation": dataset['validation']
        })
        dataset.set_format("torch", device="cuda")
        dataset_filename += f"-val_size_{val_set_size}-max_input_length_{max_input_length}"

        output_path = join(DATASET_FOLDER, dataset_filename)
        print("Saving dataset to", output_path)
        dataset.save_to_disk(output_path, num_proc=64)

if __name__ == "__main__":
    fire.Fire(main)