#!/usr/bin/env python3
"""
Train models on CIFAR-10/100 for adversarial attack experiments.

This script fine-tunes pretrained ImageNet models on CIFAR datasets.
The trained models will be used as surrogate and target models in attacks.

Usage:
    python scripts/train_models.py --dataset cifar10 --model resnet18
    python scripts/train_models.py --dataset cifar100 --model vgg16 --epochs 20
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ensemble_attacks.data import get_dataset_loaders
from ensemble_attacks.models import get_model
from ensemble_attacks.utils import get_device, set_random_seed

logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(test_loader), 100. * correct / total


def train_model(
    model_name: str,
    dataset: str,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 0.001,
    device: str = None,
    output_dir: str = "./checkpoints",
    seed: int = 42,
):
    """
    Train a model on CIFAR dataset.

    Args:
        model_name: Model architecture (resnet18, vgg16, densenet121)
        dataset: Dataset name (cifar10, cifar100)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        device: Device to use (cuda/mps/cpu or None for auto)
        output_dir: Directory to save checkpoints
        seed: Random seed
    """
    # Setup
    set_random_seed(seed)
    if device is None:
        device = get_device(prefer_cuda=False)  # Auto-detect best available device
    else:
        device = torch.device(device)

    logger.info(f"Training {model_name} on {dataset}")
    logger.info(f"Device: {device}")

    # Load dataset
    train_loader, test_loader, num_classes = get_dataset_loaders(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=4,
        device=device,
    )

    # Load model with pretrained ImageNet weights
    model = get_model(model_name, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_acc = 0.0
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint_name = f"{model_name}_{dataset}_seed{seed}.pth"
    checkpoint_path = output_path / checkpoint_name

    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Checkpoint will be saved to: {checkpoint_path}")

    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        logger.info(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✓ Saved checkpoint (best acc: {best_acc:.2f}%)")

    logger.info(f"\n{'='*80}")
    logger.info(f"Training completed!")
    logger.info(f"Best test accuracy: {best_acc:.2f}%")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info(f"{'='*80}")

    return checkpoint_path, best_acc


def main():
    parser = argparse.ArgumentParser(description="Train models for adversarial attack experiments")
    parser.add_argument("--model", type=str,
                       choices=["resnet18", "vgg16", "densenet121"],
                       help="Model architecture (required unless --train-all)")
    parser.add_argument("--dataset", type=str,
                       choices=["cifar10", "cifar100"],
                       help="Dataset to train on (required unless --train-all)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/mps/cpu, default: auto)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--train-all", action="store_true",
                       help="Train all model-dataset combinations")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.train_all:
        # Override requirement for --model and --dataset when using --train-all
        models = ["resnet18", "vgg16", "densenet121"]
        datasets = ["cifar10", "cifar100"]

        logger.info("Training all model-dataset combinations...")
        results = []

        for model in models:
            for dataset in datasets:
                logger.info(f"\n{'='*80}")
                logger.info(f"Training {model} on {dataset}")
                logger.info(f"{'='*80}")

                try:
                    checkpoint_path, best_acc = train_model(
                        model_name=model,
                        dataset=dataset,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        device=args.device,
                        output_dir=args.output_dir,
                        seed=args.seed,
                    )
                    results.append({
                        "model": model,
                        "dataset": dataset,
                        "accuracy": best_acc,
                        "checkpoint": str(checkpoint_path)
                    })
                except Exception as e:
                    logger.error(f"Failed to train {model} on {dataset}: {e}")
                    import traceback
                    traceback.print_exc()

        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        for result in results:
            logger.info(f"{result['model']:15} on {result['dataset']:10} - Acc: {result['accuracy']:.2f}%")
            logger.info(f"  Checkpoint: {result['checkpoint']}")
    else:
        # Train single model
        if not args.model or not args.dataset:
            parser.error("--model and --dataset are required when not using --train-all")

        train_model(
            model_name=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            output_dir=args.output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
