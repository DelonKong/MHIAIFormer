### For latest triplet_attention module code please refer to the corresponding file in root.
import math

import torch
import torch.nn as nn
import einops
from einops import rearrange
from torch.nn.init import trunc_normal_


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class EfficientAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, in_dims=64, token_dim=32, num_heads=2):
        super().__init__()

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)

        # w_g ->: [bs, D, 1]
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        # self.final = nn.Linear(token_dim * num_heads, token_dim)
        self.final = nn.Linear(token_dim * num_heads, token_dim * num_heads)

    def forward(self, x):
        query = self.to_query(x)  # [bs, N, D]
        key = self.to_key(x)

        # if not CoreMLConversion:
        #     # torch.nn.functional.normalize is not supported by the ANE of iPhone devices.
        #     # Using this layer improves the accuracy by ~0.1-0.2%
        #     query = torch.nn.functional.normalize(query, dim=-1)
        #     key = torch.nn.functional.normalize(key, dim=-1)

        # query_weight ->: [bs, N, 1]
        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor

        A = A.softmax(dim=-1)

        # A * query ->: [bs, N, D]
        # G ->: [bs, D]
        G = torch.sum(A * query, dim=1)

        # G ->: [bs, N, D]
        # key.shape[1] = N
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=key.shape[1]
        )

        out = self.Proj(G * key) + query

        out = self.final(out)

        return out


class Attention(nn.Module):
    """
    Vit中的Attention
    """

    def __init__(
            self,
            dim,  # 输入token的dim
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个head的dim
        self.scale = self.head_dim ** -0.5  # 缩放点积注意力中的缩放操作，即开跟操作

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 通过全连接层得到q,k,v
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # 多个head的Contact操作
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape  # [batch_size, num_patches + 1, total_embed_dim]，论文中为[batchsize, 196+1, 768]

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3（代表qkv）, num_heads（代表head数）, embed_dim_per_head（每个head的qkv维度）]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # q, k, v: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale  # 缩放

        # q: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # k.transpose(-2, -1): -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # attn: -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = q @ k.transpose(-2, -1)  # q,k 相乘，transpose为矩阵转置

        attn = attn.softmax(dim=-1)  # softmax操作
        attn = self.attn_drop(attn)  # dropout层

        x = attn @ v  # 矩阵乘法，*是元素乘法
        # x: -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]

        # x.transpose(1, 2): -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # x.transpose(1, 2).reshape(B, N, C): -> [batch_size, num_patches + 1, total_embed_dim]
        x = x.transpose(1, 2).reshape(B, N, C)  # reshape相当于把head拼接
        x = self.proj(x)  # 通过全连接进行拼接射（相当于乘论文中的Wo）
        x = self.proj_drop(x)
        return x


class SAA_Attention(nn.Module):
    def __init__(self, dim, num_heads_h=4, groups_m=8):
        super().__init__()

        self.dim = dim
        self.num_heads_h = num_heads_h
        self.groups_m = groups_m

        assert dim % num_heads_h == 0, f"dim {dim} should be divided by num_heads {num_heads_h}."
        assert dim % groups_m == 0, f"dim {dim} should be divided by num_heads {groups_m}."

        self.split_groups = self.dim // num_heads_h
        for i in range(groups_m):
            local_linear = nn.Linear(self.num_heads_h, 1)
            setattr(self, f"local_linear{i + 1}", local_linear)
        self.proj = nn.Linear(dim, dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # [bs, h, D]
        B, h, D = x.shape

        s = x.reshape(B, h, self.groups_m, -1).permute(2, 0, 3, 1)  # B, h, M, D/M ->: M, B, D/M, h
        for i in range(self.groups_m):
            local_linear = getattr(self, f"local_linear{i + 1}")
            s_i = s[i]
            s_i = local_linear(s_i)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 0)
        s_out = s_out.reshape(B, D, 1).squeeze(2)
        s_out = self.proj(s_out)
        s_out = self.sig(s_out)

        return s_out  # [bs, D]


if __name__ == "__main__":
    t = torch.randn(size=(1, 16, 64))
    out = EfficientAdditiveAttention().forward(t)
    print(out)
