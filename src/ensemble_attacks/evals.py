import torch
from tqdm import tqdm

from ensemble_attacks.attacks import (
    fgsm_attack, bim_attack, pgd_attack,
    mean_ensemble_attack, weighted_ensemble_attack
)


@torch.no_grad()
def clean_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Clean Eval"):
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total


def adversarial_accuracy(model, loader, device, attack_fn):
    model.eval()
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Adversarial Eval"):
        x, y = x.to(device), y.to(device)

        x_adv = attack_fn(model, x, y)
        preds = model(x_adv).argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    return correct / total


def evaluate_all_attacks(model, loader, device, epsilon, alpha, iters,
                         w_fgsm=0.4, w_pgd=0.3, w_bim=0.3):

    results = {}

    results["clean"] = clean_accuracy(model, loader, device)

    results["fgsm"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: fgsm_attack(m, x, y, epsilon)
    )

    results["bim"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: bim_attack(m, x, y, epsilon, alpha, iters)
    )

    results["pgd"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: pgd_attack(m, x, y, epsilon, alpha, iters)
    )

    results["mea"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: mean_ensemble_attack(m, x, y, epsilon, alpha, iters)
    )

    results["wea"] = adversarial_accuracy(
        model, loader, device,
        lambda m, x, y: weighted_ensemble_attack(
            m, x, y, epsilon, alpha, iters,
            w_fgsm=w_fgsm, w_pgd=w_pgd, w_bim=w_bim
        )
    )

    return results
