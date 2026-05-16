import torch

from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.eval import evaluate_all_attacks
from ensemble_attacks.utils import get_device


def main():
    device = get_device()
    print("Using device:", device)

    _, test_loader = get_cifar10_loaders(batch_size=128, num_workers=2)

    model = get_resnet18().to(device)

    state = torch.load("resnet18_cifar10.pt", map_location="cpu")
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    epsilon = 8 / 255
    alpha = 2 / 255
    iters = 10

    results = evaluate_all_attacks(
        model, test_loader, device,
        epsilon=epsilon, alpha=alpha, iters=iters,
        w_fgsm=0.4, w_pgd=0.3, w_bim=0.3
    )

    print("\n=== RESULTS ===")
    for k, v in results.items():
        print(f"{k:>6}: {v:.4f}")


if __name__ == "__main__":
    main()
