# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import bias_gelu_impl
from .configuration_zamba2 import Zamba2Config

class MLP(nn.Module):

    def __init__(self, config: Zamba2Config,is_expert: bool = False, layer_idx=None, num_mem_blocks = None):
        super().__init__()

        self.num_mem_blocks = num_mem_blocks
        
        self.config: Zamba2Config = config
        self.layer = layer_idx
        ffn_hidden_size_1 = self.config.ffn_hidden_size
        ffn_hidden_size_2 = self.config.ffn_hidden_size
        # If this is a gated linear unit we double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        if self.config.gated_linear_unit:
            ffn_hidden_size_1 *= 2

        if self.layer == -1:
            ffn_hidden_size_1 = 8 * self.config.hidden_size

        self.linear_fc1 = nn.Linear(self.config.hidden_size, ffn_hidden_size_1, bias = self.config.add_bias_linear)
        if self.config.gated_linear_unit or self.layer == -1:

            def glu(x):
                x = torch.chunk(x, 2, dim=-1)
                # x_ = torch.chunk(x, 4, dim=-1)
                # x = (torch.cat((x_[0], x_[2]), dim=-1), torch.cat((x_[1], x_[3]), dim=-1))

                return F.gelu(x[0]) * x[1]
            self.activation_func = glu
        else:
            self.activation_func = F.gelu


        self.linear_fc2 = nn.Linear(ffn_hidden_size_2, self.config.hidden_size, bias = self.config.add_bias_linear)
        
        # only lora the FC in here I think fc out already has a trainable projection
        if self.config.use_shared_block_lora:
            self.linear_fc1_lora_A_list = nn.ParameterList([])
            self.linear_fc1_lora_B_list = nn.ParameterList([])
            for i in range(self.num_mem_blocks):
                #linear_fc1_lora_A = nn.Parameter(torch.randn(self.config.hidden_size, self.config.lora_rank))
                linear_fc1_lora_A = nn.Linear(self.config.hidden_size, self.config.lora_rank, bias = False)
                #linear_fc1_lora_B = nn.Parameter(torch.randn(self.config.lora_rank, ffn_hidden_size_1))
                linear_fc1_lora_B = nn.Linear(self.config.lora_rank, ffn_hidden_size_1, bias = False)
                self.linear_fc1_lora_A_list.append(linear_fc1_lora_A)
                self.linear_fc1_lora_B_list.append(linear_fc1_lora_B)

    def forward(self, hidden_states, inference_params=None, forward_layer_idx = None):

        # [s, b, 4 * h/p]
        #print("IN MLP FORWARD: ", hidden_states.shape)
        if self.config.use_shared_block_lora:
            linear_fc1_lora_A = self.linear_fc1_lora_A_list[forward_layer_idx]
            linear_fc1_lora_B = self.linear_fc1_lora_B_list[forward_layer_idx]
            lora_output = linear_fc1_lora_A(hidden_states)
            lora_output= linear_fc1_lora_B(lora_output)
            intermediate_parallel = self.linear_fc1(hidden_states)
            intermediate_parallel = intermediate_parallel + lora_output
        else:
            intermediate_parallel= self.linear_fc1(hidden_states)
        
        #print("INTERMEDIATE PARALLEL: ", intermediate_parallel.shape)

        if self.config.bias_gelu_fusion:
            assert self.config.add_bias_linear is True
            assert self.activation_func == F.gelu
            intermediate_parallel = bias_gelu_impl(intermediate_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        #print("intermediate parallel prior fc2: ", intermediate_parallel.shape)
        #print("fc2_weight: ", self.linear_fc2.weight.shape)
        #print("intermediate_parallel after func", intermediate_parallel.shape)
        output = self.linear_fc2(intermediate_parallel)
        #print("MLP OUT: ", output.shape)

        return output

    def sharded_state_dict(self, prefix='', sharded_key_prefix=None, sharded_offsets=()):
        sharded_key_prefix = prefix if sharded_key_prefix is None else sharded_key_prefix
        sharded_state_dict = {}
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(
                prefix=f'{prefix}{name}.',
                sharded_key_prefix=f'{sharded_key_prefix}{name}.',
                sharded_offsets=sharded_offsets,
            )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict