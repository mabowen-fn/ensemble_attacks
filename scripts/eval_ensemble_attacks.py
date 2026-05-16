import os
import torch

from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.eval import evaluate_all_attacks
from ensemble_attacks.utils import get_device
from ensemble_attacks.logging_utils import append_csv_row, now_timestamp, ensure_dir
from ensemble_attacks.report import plot_attack_summary, plot_per_class_asr


def main():
    device = get_device()
    print("Using device:", device)

    _, test_loader = get_cifar10_loaders(batch_size=128, num_workers=2)

    model = get_resnet18()
    state = torch.load("resnet18_cifar10.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    epsilon = 8 / 255
    alpha = 2 / 255
    iters = 10

    out_dir = "outputs"
    img_dir = os.path.join(out_dir, "images")
    ensure_dir(out_dir)
    ensure_dir(img_dir)

    results = evaluate_all_attacks(
        model, test_loader, device,
        epsilon=epsilon,
        alpha=alpha,
        iters=iters,
        w_fgsm=0.4,
        w_pgd=0.3,
        w_bim=0.3,
        save_dir=img_dir
    )

    print("\n=== RESULTS ===")
    print(f"{'attack':>6} | {'acc':>8} | {'avg_linf':>10}")
    print("-" * 32)

    print(f"{'clean':>6} | {results['clean']:.4f} | {'-':>10}")

    for attack_name in ["fgsm", "bim", "pgd", "mea", "wea"]:
        acc = results[attack_name]
        linf = results[f"{attack_name}_linf"]
        print(f"{attack_name:>6} | {acc:.4f} | {linf:.6f}")

    report_dir = os.path.join(out_dir, "report")
    plot_attack_summary(results, report_dir)
    plot_per_class_asr(results, report_dir)

    print(f"Saved plots to: {report_dir}")

    # log each attack separately
    csv_path = os.path.join(out_dir, "results.csv")

    for attack_name in ["fgsm", "bim", "pgd", "mea", "wea"]:
        adv_acc = results[attack_name]
        avg_linf = results[f"{attack_name}_linf"]
        per_class_asr = results[f"{attack_name}_asr_pc"]

        row = {
            "timestamp": now_timestamp(),
            "model": "resnet18",
            "dataset": "cifar10",
            "attack": attack_name,
            "epsilon": epsilon,
            "alpha": alpha,
            "iters": iters,
            "w_fgsm": 0.4,
            "w_pgd": 0.3,
            "w_bim": 0.3,
            "clean_acc": results["clean"],
            "adv_acc": adv_acc,
            "attack_success_rate": 1.0 - adv_acc,
            "avg_linf": avg_linf,
        }

        for c in range(10):
            row[f"asr_class_{c}"] = per_class_asr[c]

        append_csv_row(csv_path, row)

    print(f"\nSaved CSV log to: {csv_path}")
    print(f"Saved adversarial images to: {img_dir}")


if __name__ == "__main__":
    main()
