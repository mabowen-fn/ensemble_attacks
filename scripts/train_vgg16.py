#!/usr/bin/env python3
"""Train VGG16 on CIFAR-10 for use as target model in adversarial attack experiments.

This script is GPU-aware and sets sensible defaults for cloud runs.
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path

from ensemble_attacks.models import get_vgg16
from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.train import train_one_epoch, evaluate
from ensemble_attacks.utils import get_device, set_random_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--model-path', type=str, default='vgg16_cifar10.pt')
    p.add_argument('--num-workers', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--deterministic', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f"Training on device: {device}")

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    weight_decay = args.weight_decay
    model_path = args.model_path

    # Load data
    print("Loading CIFAR-10...")
    try:
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            device=device,
            normalize=False,
        )
    except OSError as e:
        # Fallback for environments where many workers cause issues
        print("Data loader worker failure, falling back to num_workers=0:", e)
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=batch_size,
            num_workers=0,
            data_dir=args.data_dir,
            device=device,
            normalize=False,
        )

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

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model (test_acc={test_acc:.4f})")

    print(f"\n✓ Training complete. Best test accuracy: {best_acc:.4f}")
    print(f"✓ Model saved to {model_path}")


if __name__ == "__main__":
    main()
