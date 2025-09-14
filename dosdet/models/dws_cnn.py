from __future__ import annotations
import torch
import torch.nn as nn

class DWConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1, p=None, drop=0.0):
        super().__init__()
        if p is None:
            p = (k - 1) // 2 * d
        self.depth = nn.Conv1d(c_in, c_in, k, padding=p, dilation=d, groups=c_in)
        self.point = nn.Conv1d(c_in, c_out, 1)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.act(x)
        return self.drop(x)

class AttentionPooling(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.proj = nn.Linear(c, c)

    def forward(self, x):       # x: [B,C,M]
        x = x.transpose(1, 2)   # [B,M,C]
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn = torch.softmax((Q @ K.transpose(1,2)) / (x.size(-1) ** 0.5), dim=-1)   # [B,M,M]
        pooled = (attn @ V).mean(dim=1)   # [B,C]
        pooled = self.proj(pooled)
        weights = attn.mean(dim=1)        # [B,M]
        return pooled, weights

class FastSeqEncoder(nn.Module):
    """
    Depthwise-separable TCN-like stack with static shapes for autotuning.
    """
    def __init__(self, in_dim, channels=(64,96), k=5, drop=0.1):
        super().__init__()
        c0 = in_dim
        layers = []
        for i, c in enumerate(channels):
            d = 2 ** i
            layers.append(DWConv1d(c0, c, k=k, d=d, drop=drop))
            c0 = c
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPooling(c0)
        self.out_dim = c0

    def forward(self, x):  # x: [B,M,K] → [B,K,M]
        x = x.transpose(1, 2)
        h = self.net(x)
        pooled, w = self.pool(h)
        return pooled, w

class FastDetector(nn.Module):
    def __init__(self, seq_in_dim: int, static_dim: int, channels=(64,96), k=5, drop=0.1, mlp_hidden=(256,64), aux_family_head=True, n_families=6):
        super().__init__()
        self.seq = FastSeqEncoder(seq_in_dim, channels=channels, k=k, drop=drop)
        in_total = self.seq.out_dim + static_dim
        mlp = []
        c = in_total
        for h in mlp_hidden:
            mlp += [nn.Linear(c, h), nn.ReLU(inplace=True), nn.Dropout(drop)]
            c = h
        self.mlp = nn.Sequential(*mlp)
        self.bin_head = nn.Linear(c, 1)
        self.aux_family_head = aux_family_head
        if aux_family_head:
            self.family_head = nn.Linear(c, n_families)

    def forward(self, seq, static):
        pooled, attn = self.seq(seq)
        h = torch.cat([pooled, static], dim=-1)
        h = self.mlp(h)
        out = {"logits": self.bin_head(h), "attn": attn}
        if self.aux_family_head:
            out["family_logits"] = self.family_head(h)
        return out
