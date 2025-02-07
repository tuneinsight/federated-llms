from transformers import AutoModel

model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

# Check precision and calculate model size
param_size_bytes = 2 # for bf16
model_size_mb = (total_params * param_size_bytes) / (1024 ** 2)  # Convert bytes to MB
print(f"Approximate model size: {model_size_mb:.2f} MB")

lora_r = 16
lora_target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "o_proj", "down_proj", "gate_proj"]

global_min = float('inf')
global_max = float('-inf')

# Estimate LoRA weights
lora_params = 0
for name, param in model.named_parameters():
    if any(target in name for target in lora_target_modules):
        input_dim = param.size(0)  # assume input dim is first
        lora_params += 2 * lora_r * input_dim

        layer_min = param.data.min().item()
        layer_max = param.data.max().item()
        global_min = min(global_min, layer_min)
        global_max = max(global_max, layer_max)

print(f"LoRA parameters: {lora_params}")

# Calculate LoRA size in MB
lora_size_mb = (lora_params * param_size_bytes) / (1024 ** 2)
print(f"LoRA parameters: {lora_params}")
print(f"LoRA size: {lora_size_mb:.2f} MB")

print(f"ratio: {100*lora_params/total_params}%")
print(f"weights range: [{global_min};{global_max}]")
