from __future__ import annotations
import torch
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, seq_in_dim: int, static_dim: int, channels=(64,64,128), k=5, drop=0.15, heads=4, mlp_hidden=(128,64)):
        super().__init__()
        from .backbone import SequenceEncoder
        self.seq = SequenceEncoder(seq_in_dim, channels=channels, k=k, drop=drop, heads=heads)
        in_total = self.seq.out_dim + static_dim
        mlp = []
        c = in_total
        for h in mlp_hidden:
            mlp += [nn.Linear(c, h), nn.ReLU(inplace=True), nn.Dropout(drop)]
            c = h
        self.mlp = nn.Sequential(*mlp)
        self.bin_head = nn.Linear(c, 1)

    def forward(self, seq, static):
        # seq: [B,M,Kseq], static: [B,Kstatic]
        pooled, attn = self.seq(seq)
        h = torch.cat([pooled, static], dim=-1)
        h = self.mlp(h)
        logits = self.bin_head(h)  # [B,1]
        return {"logits": logits, "attn": attn}
