import os
import fire
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

from dp import add_noise_to_model

from typing import Tuple
import torch

from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

import gc

ASSETS = '/assets/'
MODELS = os.path.join(ASSETS, 'models/')
DATASETS = os.path.join(ASSETS, 'datasets/')

def main(
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
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = False,
    warmup_steps: int = 75,
    flash_attention: bool = True,
    resume_from_checkpoint: bool = False,
    neftune_noise_alpha=None,
    max_grad_norm=1.0,
    sigma=1.0,
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

    # Add noise to model weights, scaled by the learning rate
    add_noise_to_model(
        model=model,
        sigma=sigma,
        learning_rate=1.0, # Unused
        clipping_norm=1.0, # Unused
    )

    if use_lora:
        model = model.merge_and_unload()

    # trainer.save_model(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
