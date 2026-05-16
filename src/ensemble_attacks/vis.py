import os
import torch
from torchvision.utils import save_image


def save_adv_examples(save_dir, x_clean, x_adv, y, preds_clean, preds_adv, max_save=16):
    """
    Saves:
    - clean images
    - adversarial images
    - perturbation (scaled for visibility)
    """

    os.makedirs(save_dir, exist_ok=True)

    n = min(max_save, x_clean.size(0))

    x_clean = x_clean[:n].detach().cpu()
    x_adv = x_adv[:n].detach().cpu()

    preds_clean = preds_clean[:n].detach().cpu()
    preds_adv = preds_adv[:n].detach().cpu()
    y = y[:n].detach().cpu()

    perturb = x_adv - x_clean
    perturb_vis = (perturb - perturb.min()) / (perturb.max() - perturb.min() + 1e-8)

    save_image(x_clean, os.path.join(save_dir, "clean.png"), nrow=8)
    save_image(x_adv, os.path.join(save_dir, "adv.png"), nrow=8)
    save_image(perturb_vis, os.path.join(save_dir, "perturbation.png"), nrow=8)

    # also save labels/preds
    with open(os.path.join(save_dir, "labels.txt"), "w") as f:
        for i in range(n):
            f.write(
                f"[{i}] true={y[i].item()} clean_pred={preds_clean[i].item()} adv_pred={preds_adv[i].item()}\n"
            )
