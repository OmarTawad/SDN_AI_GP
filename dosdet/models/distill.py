from __future__ import annotations
import torch
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    KL divergence between teacher and student probabilities (binary).
    """
    T = max(1e-3, float(temperature))
    ps = torch.sigmoid(student_logits / T)
    pt = torch.sigmoid(teacher_logits / T)
    # Avoid log(0)
    eps = 1e-6
    loss = pt * torch.log((pt + eps)/(ps + eps)) + (1-pt)*torch.log(((1-pt)+eps)/((1-ps)+eps))
    return loss.mean()
