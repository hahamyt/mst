import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import numpy as np
from .topk import (batched_index_select, extract_patches_from_indicators)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer() if act_layer else nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ScoreBlock(nn.Module):
    def __init__(self, in_dim, sr_ratio=1):
        super().__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)

        self.cross_attn = CrossAttention(in_dim, sr_ratio=sr_ratio)
        self.cross_mlp = Mlp(in_features=in_dim, drop=0.0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def norm_0_1(self, x):
        return (x + 1) / 2

    def forward(self, x_b, x_s, base_idxs):
        B, n_p = base_idxs.shape
        _, pnum_s, C = x_s.shape

        x_s_, attn = self.cross_attn(self.norm1(x_s), self.norm2(x_b), return_attn=True)
        x_s = x_s + x_s_
        x_s = x_s + self.cross_mlp(self.norm3(x_s))

        # Step 1
        mask = base_idxs[:, :(n_p//2)] >= 0
        ids = base_idxs[:, :(n_p//2)].clone().long()
        ids[~mask] = 0                  # 将无点击区域的id置为0

        tokens = batched_index_select(x_b, 1, ids) * mask.unsqueeze(-1) # 提取点击区域的tokens，并将无点击区域的id置为0
        kernels = []
        # Step 2
        attn_scores = []
        for i in range(B):
            if True not in mask[i]:
                kernels.append(tokens[i].mean(dim=0, keepdim=True))
                attn_scores.append(
                    self.norm_0_1(F.cosine_similarity(x_s[i, :, :], kernels[i]).squeeze())*0.0)  # 这种计算相似度的方式配合Triplet Loss非常香（看起来）
            else:
                kernels.append(tokens[i][mask[i]].mean(dim=0, keepdim=True))
                attn_scores.append(
                    self.norm_0_1(F.cosine_similarity(x_s[i, :, :], kernels[i]).squeeze()))  # 这种计算相似度的方式配合Triplet Loss非常香（看起来）

        pos_scores = torch.stack(attn_scores)
        # Step 3
        valid_k = [min(x_s.shape[1]//8, (pos_scores[x]>0.65).sum().item()) for x in range(B)]
        k = x_s.shape[1]//8 # 使用固定尺寸的tokens交互， max(valid_k) # min(128, (pos_scores[np.random.randint(B)-1]>0.8).sum())        # 自适应选择交互的数量k
        topk_score, index = torch.topk(pos_scores, k=k)
        one_hot = F.one_hot(index, pnum_s)

        selected = torch.mul(topk_score[:, :, None].repeat([1, 1, pnum_s]), one_hot)
        binary_selected = (selected > 0.0).float()
        selected = selected + (binary_selected - selected).detach()

        return selected, index, pos_scores, x_s, torch.cat(kernels).unsqueeze(-1)

class Attention(nn.Module):
    """ Multi-head self-attention operation
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1,2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, return_attn=False):
        B, N, C = x.shape
        y = x if y is None else y
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        W = H = int(np.sqrt(y.shape[1]))
        if self.sr_ratio > 1:
            y_ = y.permute(0, 2, 1).reshape(B, C, H, W)
            y_ = self.sr(y_).reshape(B, C, -1).permute(0, 2, 1)
            y_ = self.norm(y_)
            kv = self.kv(y_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_ = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn
        else:
            return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_drop=0.0, qkv_bias=False, attn_drop=0.,
        proj_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_score=False, depth=0):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.use_score = use_score
        if self.use_score:
            self.depth = depth
            self.score_s = ScoreBlock(dim, sr_ratio=1)
            # self.score_l = ScoreBlock(dim, sr_ratio=1)

            self.norm3 = norm_layer(dim)
            self.norm4 = norm_layer(dim)
            self.score_cross_attn = CrossAttention(dim)

            self.norm5 = norm_layer(dim)
            self.score_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                   proj_drop=proj_drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

    def forward(self, x_b, x_i, idxs=None, im_sz=448, blk_depth=0):
        B, _, C = x_b.shape
        x_s, pos_emb_s = x_i[0]
        x_l, pos_emb_l = x_i[1]

        tokens = {}
        if self.use_score:
            indices_s, index_s, pos_scores_s, x_s, kernel_s = self.score_s(x_b, x_s, idxs[0])
            sel_tokens_s = extract_patches_from_indicators(x_s, indices_s) + \
                           extract_patches_from_indicators(pos_emb_s.repeat(B, 1, 1), indices_s)

            x_b_ = self.score_cross_attn(self.norm3(x_b), self.norm4(sel_tokens_s))
            x_b = x_b + x_b_
            x_b = x_b + self.score_mlp(self.norm5(x_b))

            tokens['sel_tokens_s'] = sel_tokens_s
            tokens['idx_tokens_s'] = index_s
            tokens['sel_tokens_l'] = torch.zeros_like(x_s)
            tokens['idx_tokens_l'] = torch.range(0, x_s.shape[1]-1).unsqueeze(0).repeat(B, 1).type_as(x_s).long() # index_s # torch.range(0, x_l.shape[1]-1).unsqueeze(0).repeat(B, 1).type_as(x_s).long() # index_l

            tokens['kernel_s'] = kernel_s
            tokens['kernel_l'] = torch.zeros_like(kernel_s)

            tokens['pos_scores_s'] = pos_scores_s

        x_b_ = self.attn(self.norm1(x_b))
        x_b = x_b + x_b_
        x_b = x_b + self.mlp(self.norm2(x_b))

        return x_b, [[x_s, pos_emb_s], [x_l, pos_emb_l]], tokens
