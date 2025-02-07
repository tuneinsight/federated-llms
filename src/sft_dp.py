import os
import sys
import fire
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
from datasets import load_dataset

from typing import Tuple
import torch
from trl import SFTTrainer

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

from dp import compute_sigma, add_noise_to_model

import gc

ASSETS = '/assets/'
MODELS = os.path.join(ASSETS, 'models/dp/')
DATASETS = os.path.join(ASSETS, 'datasets/')

def main(
    eps: float,
    delta: float,
    model_name: str,
    output: str,
    dataset_name: str,
    train_in_8bit: bool = False,
    max_input_length: int = 512,
    use_lora: bool = False,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    global_batch_size: int = 128,
    per_device_batch_size: int = 2,
    max_steps: int = -1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    save_total_limit: int = 30,
    eval_steps: int = 200,
    save_steps: int = 200,
    logging_steps: int = 10,
    device_map: str = "auto",
    group_by_length: bool = True,
    use_wandb: bool = False,
    wandb_project: str = "private_llm",
    optim: str = "sgd",
    lr_scheduler_type: str = "cosine",
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 75,
    flash_attention: bool = True,
    resume_from_checkpoint: bool = False,
    neftune_noise_alpha=None,
    max_grad_norm=1.0,
    **kwargs
):
    torch.cuda.empty_cache()
    gc.collect()

    output_dir = os.path.join(MODELS, output)
    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")
    
    if use_wandb and len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project

    dataset_dir = os.path.join(DATASETS, dataset_name)
    dataset = load_dataset(dataset_dir, num_proc=64)
    print(dataset)

    model_load_kwargs = {'device_map': device_map}
    if flash_attention:
        model_load_kwargs['attn_implementation'] = "flash_attention_2"

    if train_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=train_in_8bit, 
                                                llm_int8_has_fp16_weight=True,
                                                llm_int8_threshold=6)
        model_load_kwargs['torch_dtype'] = torch.bfloat16
        model_load_kwargs['quantization_config'] = quantization_config
    else:
        # loading the model with torch_dtype=torch.float16 with only fp16 and no LoRA leads
        # to `ValueError: Attempting to unscale FP16 gradients.`
        model_load_kwargs['torch_dtype'] = torch.bfloat16 if bf16 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # modules_to_save = ["lm_head", "embed_tokens"]   # because we added new tokens
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print("Initial pad token:", tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token  #  '[PAD]' tokenizer.unk_token
    tokenizer.padding_side = "left"
    print("New pad token:", tokenizer.pad_token)
    model.config.pad_token_id = tokenizer.pad_token_id

    gradient_accumulation_steps = global_batch_size // per_device_batch_size

    training_args = TrainingArguments(
        auto_find_batch_size=True,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=logging_steps,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else "none",
        run_name=output if use_wandb else "none",
        save_safetensors=False,
        neftune_noise_alpha=neftune_noise_alpha,
        max_grad_norm=max_grad_norm,
        **kwargs
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        dataset_num_proc=64,
        max_seq_length=max_input_length,
        dataset_text_field='text',
        packing=True,
        eval_packing=True,
        tokenizer=tokenizer
    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))

    if torch.__version__ >= "2" and sys.platform != "win32" and not use_lora:
        model = torch.compile(model)

    # finally, train
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if use_lora:
        model = model.merge_and_unload()

    # Computes DP parameters and add noise
    dataset_size = len(dataset['train'])
    if max_steps > 0: # If max_steps is explicitly set
        total_iterations = max_steps
    else:
        steps_per_epoch = dataset_size // global_batch_size
        total_iterations = steps_per_epoch * num_epochs

    sigma = compute_sigma(
        eps=eps,
        delta=delta,
        c2=1.1,
        batch_size=global_batch_size,
        total_samples=dataset_size,
        num_iterations = total_iterations,
        rho=1.0,
    )
    print(f"Computed sigma for gradients: {sigma:.4f}")

    # Add noise to model weights, scaled by the learning rate
    add_noise_to_model(
        model=model,
        sigma=sigma,
        learning_rate=learning_rate,
        clipping_norm=max_grad_norm,
    )

    # trainer.save_model(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
