import torch
import torch.optim as optim

from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.models import get_resnet18
from ensemble_attacks.train import train_one_epoch, evaluate
from ensemble_attacks.utils import get_device


def main():
    device = get_device()
    print("Using device:", device)

    train_loader, test_loader = get_cifar10_loaders(batch_size=128, num_workers=2)

    model = get_resnet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 11):
        loss, acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:02d} | Loss={loss:.4f} | TrainAcc={acc:.4f} | TestAcc={test_acc:.4f}")

    torch.save(model.state_dict(), "resnet18_cifar10.pt")
    print("Saved model to resnet18_cifar10.pt")


if __name__ == "__main__":
    main()
