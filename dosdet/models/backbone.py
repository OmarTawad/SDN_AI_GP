from __future__ import annotations
import torch
import torch.nn as nn

class DWConv1d(nn.Module):
    def __init__(self, c, k, d=1, p=None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2 * d
        self.depthwise = nn.Conv1d(c, c, kernel_size=k, padding=p, dilation=d, groups=c)
        self.pointwise = nn.Conv1d(c, c, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=(k - 1) // 2 * d, dilation=d),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Conv1d(c_out, c_out, k, padding=(k - 1) // 2 * d, dilation=d),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
        )
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.res(x)

class AttentionPooling(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(c, c)
        self.k = nn.Linear(c, c)
        self.v = nn.Linear(c, c)
        self.out = nn.Linear(c, c)

    def forward(self, x):
        # x: [B, C, M] → transpose to [B, M, C]
        x = x.transpose(1, 2)
        Q = self.q(x)  # [B,M,C]
        K = self.k(x)
        V = self.v(x)
        attn = torch.softmax((Q @ K.transpose(1, 2)) / (x.size(-1) ** 0.5), dim=-1)  # [B,M,M]
        pooled = attn @ V  # [B,M,C]
        pooled = pooled.mean(dim=1)  # [B,C]
        pooled = self.out(pooled)
        # For explainability: attention weights per position (mean over rows)
        weights = attn.mean(dim=1)  # [B,M]
        return pooled, weights

class SequenceEncoder(nn.Module):
    def __init__(self, in_dim, channels=(64,64,128), k=5, drop=0.15, heads=4):
        super().__init__()
        c0 = in_dim
        layers = []
        for i, c in enumerate(channels):
            d = 2 ** i
            layers.append(TCNBlock(c0, c, k=k, d=d, drop=drop))
            c0 = c
        self.net = nn.Sequential(*layers)
        self.pool = AttentionPooling(c0, heads=heads)
        self.out_dim = c0

    def forward(self, x):
        # x: [B, M, K_seq] → to [B, K_seq, M]
        x = x.transpose(1, 2)
        h = self.net(x)
        pooled, weights = self.pool(h)
        return pooled, weights  # [B,C], [B,M]
