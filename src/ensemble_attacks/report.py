import os
import matplotlib.pyplot as plt


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def plot_attack_summary(results, save_dir):
    """
    Plots:
      - adversarial accuracy per attack
      - avg Linf per attack
    """
    os.makedirs(save_dir, exist_ok=True)

    attacks = ["fgsm", "bim", "pgd", "mea", "wea"]
    accs = [results[a] for a in attacks]
    linfs = [results[f"{a}_linf"] for a in attacks]

    # Accuracy plot
    plt.figure()
    plt.bar(attacks, accs)
    plt.ylim(0, 1)
    plt.title("Adversarial Accuracy by Attack")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(save_dir, "adv_accuracy.png"))
    plt.close()

    # Linf plot
    plt.figure()
    plt.bar(attacks, linfs)
    plt.title("Average L∞ Perturbation by Attack")
    plt.ylabel("Average L∞")
    plt.savefig(os.path.join(save_dir, "avg_linf.png"))
    plt.close()


def plot_per_class_asr(results, save_dir):
    """
    For each attack, save a bar plot of ASR per class.
    """
    os.makedirs(save_dir, exist_ok=True)

    attacks = ["fgsm", "bim", "pgd", "mea", "wea"]

    for attack in attacks:
        asr = results[f"{attack}_asr_pc"]

        plt.figure(figsize=(10, 4))
        plt.bar(CIFAR10_CLASSES, asr)
        plt.ylim(0, 1)
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Per-Class Attack Success Rate (ASR) - {attack.upper()}")
        plt.ylabel("ASR")
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir, f"asr_{attack}.png"))
        plt.close()
