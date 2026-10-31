import torch
import math
from train_gpt2 import GPTConfig, GPT, write_model

# 1. Define config with native muP fields
config = GPTConfig(
        block_size=1024, 
        vocab_size=50257,
        n_layer=1,
        n_head=1,
        n_embd=64,
        use_mup=True,          # Enables native muP parameterization
        mup_width_mult=1.0     # 1.0 for hyperparameter sweeps
)

# 2. Instantiate PyTorch model (runs internal _init_weights with muP scales)
model = GPT(config)

# 3. Export to llm.c-compatible binary file
write_model(model, "my_models/mup_base_1l_64w.bin", dtype="bfloat16")

