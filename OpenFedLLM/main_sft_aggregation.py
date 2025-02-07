import copy
import os

from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args

# ===== Define the arguments =====
script_args, fed_args, peft_config = get_config()
training_args = get_training_args(script_args, 0)
save_config(script_args, fed_args)
print(script_args, fed_args)

# ===== Get local model paths =====
model_paths = script_args.model_name_or_path.split(",")
fed_args.num_clients = len(model_paths)

# ===== Split the dataset into clients =====
# sample_num_list = [1000 for _ in range(fed_args.num_clients)]
# sample_num_list = [46, 334, 185, 70]
sample_num_list = [70, 141, 293] # flashcard, medmcqa, pubmedqa

# ===== Get model config =====
device_map, quantization_config, torch_dtype = get_model_config(script_args)

models = []
for model_path in model_paths:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )

    if script_args.load_in_8bit or script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=training_args.gradient_checkpointing
                )

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    models.append(model)

# ===== Define the global and local models =====
# global_dict = copy.deepcopy(get_peft_model_state_dict(models[0]))
# local_dict_list = [copy.deepcopy(get_peft_model_state_dict(model)) for model in models]
global_dict = copy.deepcopy(models[0].state_dict())
local_dict_list = [copy.deepcopy(model.state_dict()) for model in models]

proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

clients_this_round = get_clients_this_round(fed_args, 0)

# ===== Server aggregates the local models =====
global_dict, global_auxiliary = global_aggregate(
    fed_args, global_dict, local_dict_list, sample_num_list, \
    clients_this_round, 0, proxy_dict=proxy_dict, \
    opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict)
)

# set_peft_model_state_dict(models[0], global_dict) # Update global model
models[0].load_state_dict(global_dict) # Update global model

# ===== Save the model =====
output_dir = os.path.abspath(os.getenv('MERGED_OUTPUT', ""))
output_dir = output_dir or script_args.output_dir
output_dir = os.path.abspath(output_dir)

merged_path_adapter = os.path.join(output_dir, "adapter")
merged_path_full_model = os.path.join(output_dir)

# models[0].save_pretrained(merged_path_adapter) # Save adapter
models[0].save_pretrained(merged_path_full_model) # Save full model

print(merged_path_full_model)
