from .mamba_model import MambaModel
from .mamba_config import MambaConfig
import torch

config = MambaConfig(
    num_layers = 5,
    hidden_size = 1024,
    mamba_headdim = 64,
    mamba_ngroups = 1,
    state_size = 16,
    conv_dimension = 4,
    expansion_factor = 2,
    rms_norm = True,
    bias = False,
    use_mem_mlp = True,
    num_attention_heads = 16,
    num_mem_heads = 16,
    num_mem_blocks = 2,
    vocab_size = 50000,
    layer_mapping = ["m", "m", "g", "m", "m"]
    
)
#layer_mapping = ["r", "r", "g", "r", "r"]
model = MambaModel(config = config, max_sequence_length = 4096)
model = model.cuda().half()
inputs = torch.tensor([1, 2]).cuda().long().unsqueeze(0)
out = model(inputs)
print("OUT", out)