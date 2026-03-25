import torch
import os
import gc
from litgpt import LLM
from pathlib import Path

checkpoint_model = "microsoft/Phi-3-mini-128k-instruct"
checkpoint_model_checker = "microsoft/Phi-3-mini-128k-instruct"

# \XXX is the path
base_path = Path(r"C:\XXX\checkpoints")
checkpoint_path = base_path / checkpoint_model
checkpoint_path_checker = base_path / checkpoint_model_checker

output_dir = Path(r"C:\XXX")
output_file = output_dir /"llm_model_weights.pth"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Loading model from: {checkpoint_path}")

output_dir_checker = Path(r"C:\XXX")
output_file_checker = output_dir_checker /"llm_model_weights_checker.pth"
output_dir_checker.mkdir(parents=True, exist_ok=True)
print(f"Loading model from: {checkpoint_path_checker}")

try:
    print(f"Loading first model: {checkpoint_path}")
    llm = LLM.load(checkpoint_path)
    llm.model.to("cpu")
    torch.save(llm.model.state_dict(), output_file)
    print(f"Successfully serialized weights to: {output_file}")

    del llm
    gc.collect() 
    torch.cuda.empty_cache() 

    print(f"Loading second model: {checkpoint_path_checker}")
    llm_checker = LLM.load(checkpoint_path_checker)
    llm_checker.model.to("cpu")
    torch.save(llm_checker.model.state_dict(), output_file_checker)
    print(f"Successfully serialized weights to: {output_file_checker}")


    del llm_checker
    gc.collect()

except FileNotFoundError as e:
    print(f"\n[ERROR]: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
