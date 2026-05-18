import torch
from tqdm import tqdm

from ensemble_attacks.attacks import (
    fgsm_attack, bim_attack, pgd_attack,
    mean_ensemble_attack, weighted_ensemble_attack
)

from ensemble_attacks.vis import save_adv_examples


@torch.no_grad()
def clean_accuracy(model, loader, device, log_fn=None):
    """
    Compute accuracy on clean (unperturbed) data.
    
    Args:
        model: Neural network model
        loader: DataLoader for test data
        device: torch.device to use
        log_fn: Optional logging function
        
    Returns:
        float: Accuracy on clean data
    """
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Clean Eval", leave=False):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        assert x.min() >= 0 and x.max() <= 1, "Input must be in [0, 1] range"

        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    
    accuracy = correct / total if total > 0 else 0
    
    if log_fn:
        log_fn(f"Clean accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy


def adversarial_accuracy(model, loader, device, attack_fn, num_classes=10, save_images_dir=None, log_fn=None):
    """
    Compute adversarial accuracy and metrics.
    
    Returns:
      Tuple[float, float, list]: 
        - adv_acc: Adversarial accuracy (fraction of correctly classified adversarial samples)
        - avg_linf: Average L∞ perturbation magnitude
        - per_class_asr: Per-class attack success rates (only on clean-correct samples)
        
    per_class_asr[c] =
        (# samples of class c that were clean-correct AND become wrong after attack)
        / (# samples of class c that were clean-correct)
    """
    model.eval()

    correct = 0
    total = 0

    linf_sum = 0.0
    linf_count = 0

    # Per-class stats
    clean_correct_count = torch.zeros(num_classes, dtype=torch.long)
    attack_success_count = torch.zeros(num_classes, dtype=torch.long)

    saved = False

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Adversarial Eval", leave=False)):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        assert x.min() >= 0 and x.max() <= 1, "Input must be in [0, 1] range"

        # Clean predictions
        with torch.no_grad():
            preds_clean = model(x).argmax(dim=1)

        # Generate adversarial examples
        x_adv = attack_fn(model, x, y)
        
        # Validate adversarial examples
        assert x_adv.min() >= 0 and x_adv.max() <= 1, "Adversarial samples out of [0, 1] range"

        # Adversarial predictions
        with torch.no_grad():
            preds_adv = model(x_adv).argmax(dim=1)

        # Adversarial accuracy
        correct += (preds_adv == y).sum().item()
        total += x.size(0)

        # Compute L∞ perturbation
        delta = (x_adv - x).abs()
        batch_linf = delta.view(delta.size(0), -1).max(dim=1).values  # (batch,)
        linf_sum += batch_linf.sum().item()
        linf_count += batch_linf.size(0)

        # Per-class ASR calculation
        clean_correct_mask = (preds_clean == y)
        attack_success_mask = clean_correct_mask & (preds_adv != y)

        for c in range(num_classes):
            class_mask = (y == c)
            clean_correct_count[c] += (clean_correct_mask & class_mask).sum().item()
            attack_success_count[c] += (attack_success_mask & class_mask).sum().item()

        # Save first batch images only
        if (save_images_dir is not None) and (not saved):
            save_adv_examples(
                save_images_dir,
                x_clean=x,
                x_adv=x_adv,
                y=y,
                preds_clean=preds_clean,
                preds_adv=preds_adv,
                max_save=16
            )
            saved = True

    adv_acc = correct / total if total > 0 else 0
    avg_linf = linf_sum / linf_count if linf_count > 0 else 0

    per_class_asr = []
    for c in range(num_classes):
        denom = clean_correct_count[c].item()
        if denom == 0:
            per_class_asr.append(0.0)
        else:
            per_class_asr.append(attack_success_count[c].item() / denom)

    if log_fn:
        log_fn(f"Adversarial accuracy: {adv_acc:.4f}, Avg L∞: {avg_linf:.6f}")
    
    return adv_acc, avg_linf, per_class_asr


def evaluate_all_attacks(model, loader, device, epsilon, alpha, iters,
                         w_fgsm=0.4, w_pgd=0.3, w_bim=0.3,
                         save_dir=None, log_fn=None):
    """
    Evaluate all attack types and log results.
    
    Returns:
        dict: Results dictionary with accuracy and metrics for each attack type
    """
    assert abs((w_fgsm + w_pgd + w_bim) - 1.0) < 1e-6, "Weights must sum to 1.0"
    
    results = {}

    if log_fn:
        log_fn("Evaluating clean accuracy...")
    results["clean"] = clean_accuracy(model, loader, device, log_fn=log_fn)

    # FGSM
    if log_fn:
        log_fn("Evaluating FGSM attack...")
    results["fgsm"], results["fgsm_linf"], results["fgsm_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: fgsm_attack(m, x, y, epsilon),
        save_images_dir=None if save_dir is None else f"{save_dir}/fgsm",
        log_fn=log_fn
    )

    # BIM
    if log_fn:
        log_fn("Evaluating BIM attack...")
    results["bim"], results["bim_linf"], results["bim_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: bim_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/bim",
        log_fn=log_fn
    )

    # PGD
    if log_fn:
        log_fn("Evaluating PGD attack...")
    results["pgd"], results["pgd_linf"], results["pgd_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: pgd_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/pgd",
        log_fn=log_fn
    )

    # MEA
    if log_fn:
        log_fn("Evaluating Mean Ensemble Attack (MEA)...")
    results["mea"], results["mea_linf"], results["mea_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: mean_ensemble_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/mea",
        log_fn=log_fn
    )

    # WEA
    if log_fn:
        log_fn("Evaluating Weighted Ensemble Attack (WEA)...")
    results["wea"], results["wea_linf"], results["wea_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: weighted_ensemble_attack(
            m, x, y, epsilon, alpha, iters,
            w_fgsm=w_fgsm, w_pgd=w_pgd, w_bim=w_bim
        ),
        save_images_dir=None if save_dir is None else f"{save_dir}/wea",
        log_fn=log_fn
    )

    return results
