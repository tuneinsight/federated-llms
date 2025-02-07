"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH] --output [OUTPUT-MODEL-PATH]
"""

import os

import argparse
import torch

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer
)
from peft import (
    PeftModel
)

ASSETS  = '/assets/'
MODELS = os.path.join(ASSETS, 'models/')

def main(
    base_model: str,
    lora_model: str,
    output: str,
    train_in_8bit: bool = False,
    flash_attention: bool = True,
    device_map: str = "auto",
    bf16: bool = True,
    gradient_checkpointing: bool = False
):
    use_lora = True
    output_dir = os.path.join(MODELS, output)
    lora_model = os.path.join(MODELS, lora_model)
    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if use_lora and gradient_checkpointing:
        raise ValueError("gradient_checkpointing with LoRA training is not implemented")
    
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
    base_model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(lora_model, use_fast=False)
    # print("Initial pad token:", tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token  #  '[PAD]' tokenizer.unk_token
    tokenizer.padding_side = "left"
    # print("New pad token:", tokenizer.pad_token)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model_to_merge = PeftModel.from_pretrained(base_model, lora_model)
    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=False)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(args.base_model_path, args.lora_path, args.output)
