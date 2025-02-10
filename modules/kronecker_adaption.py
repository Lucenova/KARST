# -*- coding: utf-8 -*-
"""
# @project    : KARST
# @author     : https://github.com/Lucenova
# @date       : 2025/2/10 11:05
# @brief      : 
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import factorization, make_kron

class KroneckerAdaption(nn.Module):
    def __init__(
            self,
            original_module: nn.Module,
            multiplier: float = 1.0,
            lora_dim: int = 4,
            alpha: int = 1,
            rank_dropout: float = 0.0,
            factor: int = -1,
            kronecker_w2_a_dims: list = None,
            use_scalar=False,
            rank_dropout_scale: bool = False,
            weight_decompose: bool = False,
            group_num: int = 2,
    ):
        super().__init__()

        in_dim = original_module.in_features
        out_dim = original_module.out_features
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        shape = ((out_l, out_k), (in_m, in_n))

        self.shape = (out_dim, in_dim)

        self.w1_group_1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))
        if lora_dim < max(shape[0][1], shape[1][1]):
            # bigger part. weight and LoRA. [b, dim] x [dim, d]
            # w1_a ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            # w2_a ⊗ (w2_a x w2_b) = (a, b)⊗((c, dim)x(dim, d)) = (a, b)⊗(c, d) = (ac, bd)
            if kronecker_w2_a_dims is not None:
                self.w2_group_a = nn.ParameterList(
                    nn.ParameterList([torch.empty(self.lora_dim, dim) for dim in kronecker_w2_a_dims]) for _
                    in range(group_num)
                )
            else:
                self.w2_group_a = nn.ParameterList(
                    nn.Parameter(torch.empty(self.lora_dim, shape[0][1])) for _ in range(group_num)
                )
            self.w2_group_b = nn.ParameterList(
                nn.Parameter(torch.empty(shape[1][1], self.lora_dim)) for _ in range(group_num))

        else:
            raise f'lora_dim {lora_dim} is too large for weight shape {shape[0][1], shape[1][1]}'

        self.operator = F.linear
        self.weight_decompose = weight_decompose
        if self.weight_decompose:
            original_weight: nn.Parameter = original_module.weight
            self.dora_norm_dims = original_weight.dim() - 1
            self.dora_scale = nn.Parameter(
                torch.norm(original_weight.transpose(1, 0).reshape(original_weight.shape[1], -1), dim=1, keepdim=True)
                .reshape(original_weight.shape[1], *[1] * self.dora_norm_dims).transpose(1, 0)
            ).float()

        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale

        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = torch.tensor(1.0)

        # init w1_group_1, w1_group_2
        for i in range(self.group_num):
            torch.nn.init.kaiming_uniform_(self.w1_group[i], a=math.sqrt(5))

        for w2_group_i_a in self.w2_group_a:
            if isinstance(w2_group_i_a, nn.ParameterList):
                for param in w2_group_i_a:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            else:
                torch.nn.init.kaiming_uniform_(w2_group_i_a, a=math.sqrt(5))

        for w2_group_i_b in self.w2_group_b:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(w2_group_i_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(w2_group_i_b, 0)

        self.multiplier = multiplier
        self.org_module = [original_module]
        self.org_forward = self.org_module[0].forward

    def apply_weight_decompose(self, weight):
        weight_norm = (
            weight.transpose(0, 1)
            .reshape(weight.shape[1], -1)
            .norm(dim=1, keepdim=True)
            .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
            .transpose(0, 1)
        )

        return weight * (self.dora_scale / weight_norm)

    def forward(self, x: torch.Tensor):
        delta_weight_list = []
        for w1_group_i, w2_group_i_a, w2_group_i_b in zip(self.w1_group, self.w2_group_a, self.w2_group_b):
            if isinstance(w2_group_i_a, nn.ParameterList):
                delta_weight_i = torch.concat(
                    [make_kron(w1_group_i, p.T @ w2_group_i_b.T, torch.tensor(self.multiplier * self.scale)) for p in
                     w2_group_i_a], dim=0
                )
            else:
                delta_weight_i = make_kron(
                    w1_group_i,
                    w2_group_i_a.T @ w2_group_i_b.T,
                    torch.tensor(self.multiplier * self.scale),
                )
            delta_weight_list.append(delta_weight_i)

        delta_weight = torch.sum(torch.stack(delta_weight_list), dim=0)
        # delta_weight = delta_weight / 2

        if self.shape is not None:
            delta_weight = delta_weight.reshape(self.shape)

        if self.training and self.rank_dropout:
            drop = (torch.rand(delta_weight.size(0)) > self.rank_dropout).to(delta_weight.dtype)
            drop = drop.view(-1, *[1] * len(delta_weight.shape[1:])).to(delta_weight.device)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            delta_weight *= drop

        delta_weight = delta_weight * self.scalar * self.multiplier

        weight = self.org_module[0].weight.data.to(x.device, dtype=next(self.parameters()).dtype) + delta_weight
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data

        if self.weight_decompose:
            weight = weight * self.dora_scale

        return self.operator(x, weight.view(self.shape), bias)