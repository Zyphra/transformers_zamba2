# coding=utf-8
# Copyright 2024 Zyphra Technologies and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Zamba model configuration"""

import math

from ...configuration_utils import PretrainedConfig
from ...utils import logging

import torch.nn.functional as F


logger = logging.get_logger(__name__)


class Zamba2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Zamba2Model`]. It is used to instantiate a
    Zamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Zamba-v2 model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Zamba model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ZambaModel`]
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        hidden_size (`int`, *optional*, defaults to 3712):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 14848):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 76):
            Number of hidden layers in the model.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=None`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf).
        n_mamba_heads (`<fill_type>`, *optional*, defaults to 2): <fill_docstring>
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the decoder.
        hidden_mamba_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the mamba layer.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        num_logits_to_keep (`int` or `None`, *optional*, defaults to 1):
            Number of prompt logits to calculate during generation. If `None`, all logits will be calculated. If an
            integer value, only last `num_logits_to_keep` logits will be calculated. Default is 1 because only the
            logits of the last prompt token are needed for generation. For long sequences, the logits for the entire
            sequence may use a lot of memory so, setting `num_logits_to_keep=1` will reduce memory footprint
            significantly.
        pad_token_id (`int`, *optional*, defaults to 0):
            The id of the padding token.
        bos_token_id (`int`, *optional*, defaults to 1):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 2):
            The id of the "end-of-sequence" token.
        unk_token_id (`<fill_type>`, *optional*, defaults to 0): <fill_docstring>
        sliding_window (`int`, *optional*):
            Sliding window attention window size. If not specified, will default to `None`.
        max_position_embeddings (`int`, *optional*, defaults to 4096):
            This value doesn't have any real effect. The maximum sequence length that this model is intended to be
            used with. It can be used with longer sequences, but performance may degrade.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        attn_layer_period (`int`, *optional*, defaults to 6):
            Once in this many layers, we will have a shared attention layer
        attn_layer_offset (`int`, *optional*, defaults to 4):
            Offset of the shared attention layer
        use_mamba_kernels (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use the fast mamba kernels. These are available only if `mamba-ssm` and
            `causal-conv1d` are installed, and the mamba modules are running on a CUDA device. Raises ValueError if
            `True` and kernels are not available
        mamba_d_state (`int`, *optional*, defaults to 16):
            The dimension the mamba state space latents
        mamba_d_conv (`int`, *optional*, defaults to 4):
            The size of the mamba convolution kernel
        mamba_expand (`int`, *optional*, defaults to 2):
            Expanding factor (relative to hidden_size) used to determine the mamba intermediate size
        mamba_dt_rank (`Union[int,str]`, *optional*, defaults to `"auto"`):
            Rank of the the mamba discretization projection matrix. `"auto"` means that it will default to `math.ceil(self.hidden_size / 16)`
        mamba_conv_bias (`bool`, *optional*, defaults to `True`):
            Flag indicating whether or not to use bias in the convolution layer of the mamba mixer block.
        mamba_proj_bias (`bool`, *optional*, defaults to `False`):
            Flag indicating whether or not to use bias in the input and output projections (["in_proj", "out_proj"]) of the mamba mixer block
        rope_theta (`<fill_type>`, *optional*, defaults to 10000): <fill_docstring>

    """

    model_type = "zamba2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,
        tie_word_embeddings=True,
        hidden_size=2560,
        state_size=64,
        conv_dimension=4,
        expansion_factor=2,
        use_low_rank_mamba_proj=False,
        add_bias_linear=False,
        mamba_headdim=64,
        ffn_hidden_size=None,
        gated_linear_unit=True,
        bias_gelu_fusion=False,
        lora_rank=128,
        num_hidden_layers=54,
        num_attention_heads=32,
        num_key_value_heads=None,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        num_logits_to_keep=1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        sliding_window=None,
        attention_dropout=0.0,
        use_shared_block_lora=True,
        use_mamba_kernels=True,
        **kwargs,
    ):

        self.vocab_size = vocab_size
        self.tie_word_embeddings = tie_word_embeddings
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.state_size = state_size
        self.conv_dimension = conv_dimension
        self.expansion_factor = expansion_factor
        self.use_low_rank_mamba_proj = use_low_rank_mamba_proj
        self.add_bias_linear = add_bias_linear
        self.mamba_headdim = mamba_headdim
        self.gated_linear_unit = gated_linear_unit
        self.use_shared_block_lora = use_shared_block_lora
        self.lora_rank = lora_rank
        
        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.num_attention_heads = num_attention_heads
        self.kv_channels = self.hidden_size // self.num_attention_heads
        self.num_query_groups = self.num_attention_heads
        self.bias_gelu_fusion = bias_gelu_fusion
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        if ffn_hidden_size is None:
            self.ffn_hidden_size = 4 * self.hidden_size

        self.use_cache = use_cache
        self.num_logits_to_keep = num_logits_to_keep
        self.use_mamba_kernels = use_mamba_kernels


        # Below, "m" means mamba layer, "g" means shared transformer layer followed by a mamba layer
        self.layers_block_type = ['m', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'm', 'g', 'm', 'm', 'm', 'g', 'm', 'm']

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

