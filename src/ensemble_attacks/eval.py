import torch
from tqdm import tqdm

from ensemble_attacks.attacks import (
    fgsm_attack, bim_attack, pgd_attack,
    mean_ensemble_attack, weighted_ensemble_attack
)

from ensemble_attacks.vis import save_adv_examples


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Clean Eval"):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)

        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total


def adversarial_accuracy(model, loader, device, attack_fn, num_classes=10, save_images_dir=None):
    """
    Returns:
      adv_acc (float)
      avg_linf (float)
      per_class_asr (list[float] length num_classes)

    per_class_asr[c] =
        (# samples of class c that were clean-correct AND become wrong after attack)
        / (# samples of class c that were clean-correct)
    """
    model.eval()

    correct = 0
    total = 0

    linf_sum = 0.0
    linf_count = 0

    # per-class stats
    clean_correct_count = torch.zeros(num_classes, dtype=torch.long)
    attack_success_count = torch.zeros(num_classes, dtype=torch.long)

    saved = False

    for x, y in tqdm(loader, desc="Adversarial Eval"):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device)

        # clean preds
        with torch.no_grad():
            preds_clean = model(x).argmax(dim=1)

        # generate adversarial examples
        x_adv = attack_fn(model, x, y)

        # adversarial preds
        with torch.no_grad():
            preds_adv = model(x_adv).argmax(dim=1)

        # adversarial accuracy
        correct += (preds_adv == y).sum().item()
        total += x.size(0)

        # compute L∞ perturbation
        delta = (x_adv - x).abs()
        batch_linf = delta.view(delta.size(0), -1).max(dim=1).values  # (batch,)
        linf_sum += batch_linf.sum().item()
        linf_count += batch_linf.size(0)

        # per-class ASR calculation
        clean_correct_mask = (preds_clean == y)
        attack_success_mask = clean_correct_mask & (preds_adv != y)

        for c in range(num_classes):
            class_mask = (y == c)
            clean_correct_count[c] += (clean_correct_mask & class_mask).sum().item()
            attack_success_count[c] += (attack_success_mask & class_mask).sum().item()

        # save first batch images only
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

    adv_acc = correct / total
    avg_linf = linf_sum / linf_count

    per_class_asr = []
    for c in range(num_classes):
        denom = clean_correct_count[c].item()
        if denom == 0:
            per_class_asr.append(0.0)
        else:
            per_class_asr.append(attack_success_count[c].item() / denom)

    return adv_acc, avg_linf, per_class_asr


def evaluate_all_attacks(model, loader, device, epsilon, alpha, iters,
                         w_fgsm=0.4, w_pgd=0.3, w_bim=0.3,
                         save_dir=None):

    results = {}

    results["clean"] = clean_accuracy(model, loader, device)

    # FGSM
    results["fgsm"], results["fgsm_linf"], results["fgsm_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: fgsm_attack(m, x, y, epsilon),
        save_images_dir=None if save_dir is None else f"{save_dir}/fgsm"
    )

    # BIM
    results["bim"], results["bim_linf"], results["bim_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: bim_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/bim"
    )

    # PGD
    results["pgd"], results["pgd_linf"], results["pgd_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: pgd_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/pgd"
    )

    # MEA
    results["mea"], results["mea_linf"], results["mea_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: mean_ensemble_attack(m, x, y, epsilon, alpha, iters),
        save_images_dir=None if save_dir is None else f"{save_dir}/mea"
    )

    # WEA
    results["wea"], results["wea_linf"], results["wea_asr_pc"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: weighted_ensemble_attack(
            m, x, y, epsilon, alpha, iters,
            w_fgsm=w_fgsm, w_pgd=w_pgd, w_bim=w_bim
        ),
        save_images_dir=None if save_dir is None else f"{save_dir}/wea"
    )

    return results
