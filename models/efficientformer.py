"""
EfficientFormer_v2
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict
import itertools

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.layers.helpers import to_2tuple
        
class Attention4D(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=8,
                 act_layer=nn.ReLU,
                 stride=None, norm = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads

        if stride is not None:
            self.resolution = math.ceil(resolution / stride)
            if norm:
                self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),
                                                nn.BatchNorm2d(dim), )
            else:
                self.stride_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim),)
            self.upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
            self.stride = stride
        else:
            self.resolution = resolution
            self.stride_conv = None
            self.upsample = None
            self.stride = 1

        self.N = self.resolution ** 2
        self.N2 = self.N
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        if norm:
            self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                                nn.BatchNorm2d(self.num_heads * self.key_dim), )
            self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),
                                nn.BatchNorm2d(self.num_heads * self.key_dim), )
            self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),
                                nn.BatchNorm2d(self.num_heads * self.d),
                                )
            self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                                kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),
                                        nn.BatchNorm2d(self.num_heads * self.d), )
        else:
            self.q = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),)
            self.k = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.key_dim, 1),)
            self.v = nn.Sequential(nn.Conv2d(dim, self.num_heads * self.d, 1),)
            self.v_local = nn.Sequential(nn.Conv2d(self.num_heads * self.d, self.num_heads * self.d,
                                                kernel_size=3, stride=1, padding=1, groups=self.num_heads * self.d),)
        self.talking_head1 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)
        self.talking_head2 = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1, padding=0)

        if norm:
            self.proj = nn.Sequential(act_layer(),
                                    nn.Conv2d(self.dh, dim, 1),
                                    nn.BatchNorm2d(dim), )
        else:
            self.proj = nn.Sequential(act_layer(),
                                    nn.Conv2d(self.dh, dim, 1),)

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)
        k = self.k(x).flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 2, 3)
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, self.N).permute(0, 1, 3, 2)

        attn = (
                (q @ k) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        # attn = (q @ k) * self.scale
        attn = self.talking_head1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v)

        out = x.transpose(2, 3).reshape(B, self.dh, H//self.stride, W//self.stride) + v_local
        if self.upsample is not None:
            out = self.upsample(out)

        out = self.proj(out)
        return out

class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0., mid_conv=False,norm=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mid_conv = mid_conv
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
        self.norm = norm 

        if self.mid_conv:
            if norm:
                self.mid1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features)
                self.mid_norm1 = nn.BatchNorm2d(hidden_features)
            else:
                self.mid1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                    groups=hidden_features)
    
        if norm:
            self.norm1 = nn.BatchNorm2d(hidden_features)
            self.norm2 = nn.BatchNorm2d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        if self.norm:
            x = self.norm1(x)
        x = self.act(x)

        if self.mid_conv:
            x_mid = self.mid1(x)
            if self.norm:
                x_mid = self.mid_norm1(x_mid)
            x = self.act(x_mid)

        x = self.drop(x)

        x = self.fc2(x)
        if self.norm:
            x = self.norm2(x)

        x = self.drop(x)
        return x
    
class AttnFFN(nn.Module):
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.ReLU,
                 resolution=8, stride=None,norm=False):

        super().__init__()

        self.token_mixer = Attention4D(dim, resolution=resolution, act_layer=act_layer, stride=stride,norm=norm)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, mid_conv=True,norm = not (norm))

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.mlp(x)
        return x
        
class FFN(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,norm = False):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop, mid_conv=True,norm = norm)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
           
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_2 * self.mlp(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x
