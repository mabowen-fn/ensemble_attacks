import torch

def linf_norm(x_adv, x_clean):
    """
    Returns per-sample L∞ norm.
    Shape: (batch,)
    """
    delta = (x_adv - x_clean).abs()
    return delta.view(delta.size(0), -1).max(dim=1).values
