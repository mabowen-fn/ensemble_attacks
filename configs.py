#!/usr/bin/env python3
"""
Configuration templates for large-scale experiments.
Usage: modify one of these configs and pass it to the experiment script.
"""

from dataclasses import asdict
from scripts.main_large_scale_experiment import ExperimentConfig
import json
from pathlib import Path


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Save configuration to JSON file."""
    data = asdict(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✓ Config saved to {path}")


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return ExperimentConfig(**data)


# Pre-defined configurations

QUICK_TEST = ExperimentConfig(
    seed=42,
    datasets=["cifar10"],
    surrogate_model="resnet18",
    target_model="vgg16",
    num_samples={"cifar10": 10},
    epsilon=8.0 / 255.0,
    step_size=2.0 / 255.0,
    num_steps=10,
    query_budget=100,
    elpd_method="waic",
    min_samples_for_waic=6,
    n_nes_samples=4,
    nes_sigma=1.5e-2,
    eta_ema_alpha=0.4,
    use_pretrained=True,  # Use ImageNet pretrained weights
    output_dir="./outputs/quick_test",
)

MEDIUM_RUN = ExperimentConfig(
    seed=42,
    datasets=["cifar10"],
    surrogate_model="resnet18",
    target_model="vgg16",
    num_samples={"cifar10": 50},
    epsilon=8.0 / 255.0,
    step_size=2.0 / 255.0,
    num_steps=15,
    query_budget=500,
    elpd_method="waic",
    min_samples_for_waic=6,
    n_nes_samples=8,
    nes_sigma=1.5e-2,
    eta_ema_alpha=0.4,
    use_pretrained=True,
    output_dir="./outputs/medium_run",
)

FULL_EXPERIMENT = ExperimentConfig(
    seed=42,
    datasets=["cifar10", "cifar100"],
    surrogate_model="resnet18",
    target_model="vgg16",
    num_samples={"cifar10": 100, "cifar100": 200},
    epsilon=8.0 / 255.0,
    step_size=2.0 / 255.0,
    num_steps=25,
    query_budget=1000,
    elpd_method="waic",
    min_samples_for_waic=6,
    n_nes_samples=8,
    nes_sigma=1.5e-2,
    eta_ema_alpha=0.4,
    use_pretrained=True,
    output_dir="./outputs/full_experiment",
)

PUBLICATION_READY = ExperimentConfig(
    seed=42,
    datasets=["cifar10", "cifar100"],
    surrogate_model="resnet18",
    target_model="vgg16",
    num_samples={"cifar10": 500, "cifar100": 500},
    epsilon=8.0 / 255.0,
    step_size=2.0 / 255.0,
    num_steps=30,
    query_budget=2000,
    elpd_method="waic",
    min_samples_for_waic=6,
    n_nes_samples=8,
    nes_sigma=1.5e-2,
    eta_ema_alpha=0.4,
    use_pretrained=True,
    output_dir="./outputs/publication_ready",
)

CROSS_ARCHITECTURE = ExperimentConfig(
    seed=42,
    datasets=["cifar10"],
    surrogate_model="resnet18",
    target_model="densenet121",  # Test different target architecture
    num_samples={"cifar10": 100},
    epsilon=8.0 / 255.0,
    step_size=2.0 / 255.0,
    num_steps=25,
    query_budget=1000,
    elpd_method="waic",
    min_samples_for_waic=6,
    n_nes_samples=8,
    nes_sigma=1.5e-2,
    eta_ema_alpha=0.4,
    use_pretrained=True,
    output_dir="./outputs/cross_architecture",
)


if __name__ == "__main__":
    # Save all configs to files for reference
    configs = {
        "quick_test": QUICK_TEST,
        "medium_run": MEDIUM_RUN,
        "full_experiment": FULL_EXPERIMENT,
        "publication_ready": PUBLICATION_READY,
        "cross_architecture": CROSS_ARCHITECTURE,
    }
    
    config_dir = Path("configs/experiments")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    for name, cfg in configs.items():
        save_config(cfg, config_dir / f"{name}.json")
    
    print(f"\n✓ All configs saved to {config_dir}/")
    print("\nUsage:")
    print("  from configs import load_config")
    print("  cfg = load_config('configs/experiments/quick_test.json')")
