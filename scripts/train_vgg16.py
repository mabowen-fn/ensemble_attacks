#!/usr/bin/env python3
"""Train VGG16 on CIFAR-10 for use as target model in adversarial attack experiments."""

import torch
import torch.optim as optim
from pathlib import Path

from ensemble_attacks.models import get_vgg16
from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.train import train_one_epoch, evaluate
from ensemble_attacks.utils import get_device


def main():
    device = get_device()
    print(f"Training on device: {device}")

    # Hyperparameters
    epochs = 100
    batch_size = 128
    learning_rate = 0.01
    weight_decay = 5e-4
    model_path = "vgg16_cifar10.pt"

    # Load data
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size, num_workers=4)

    # Create model
    print("Creating VGG16 model...")
    model = get_vgg16(num_classes=10)
    model.to(device)

    # Optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model (test_acc={test_acc:.4f})")

    print(f"\n✓ Training complete. Best test accuracy: {best_acc:.4f}")
    print(f"✓ Model saved to {model_path}")


if __name__ == "__main__":
    main()
