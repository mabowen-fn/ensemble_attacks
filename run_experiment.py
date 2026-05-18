#!/usr/bin/env python3
"""
Command-line interface for running large-scale experiments.

Usage:
    python run_experiment.py                    # Run default (quick test)
    python run_experiment.py --config medium    # Run medium experiment
    python run_experiment.py --list-configs     # Show available configs
    python run_experiment.py --config custom.json
"""

import argparse
import sys
from pathlib import Path
from dataclasses import asdict
import json

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scripts.main_large_scale_experiment import ExperimentConfig, run_experiment
from configs import QUICK_TEST, MEDIUM_RUN, FULL_EXPERIMENT, PUBLICATION_READY, CROSS_ARCHITECTURE


CONFIGS = {
    "quick": QUICK_TEST,
    "quick-test": QUICK_TEST,
    "medium": MEDIUM_RUN,
    "medium-run": MEDIUM_RUN,
    "full": FULL_EXPERIMENT,
    "full-experiment": FULL_EXPERIMENT,
    "publication": PUBLICATION_READY,
    "publication-ready": PUBLICATION_READY,
    "cross-arch": CROSS_ARCHITECTURE,
    "cross-architecture": CROSS_ARCHITECTURE,
}


def load_config_from_file(path: str) -> ExperimentConfig:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return ExperimentConfig(**data)


def print_config_summary(cfg: ExperimentConfig):
    """Print a nice summary of the experiment configuration."""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    print(f"Datasets:         {', '.join(cfg.datasets)}")
    print(f"Samples:          {cfg.num_samples}")
    print(f"Surrogate:        {cfg.surrogate_model}")
    print(f"Target:           {cfg.target_model}")
    print(f"Epsilon:          {cfg.epsilon:.6f}")
    print(f"Query Budget:     {cfg.query_budget}")
    print(f"Steps:            {cfg.num_steps}")
    print(f"Device:           {cfg.device_str or 'auto-detect'}")
    print(f"Output Dir:       {cfg.output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Large-scale power likelihood attacks experiment runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="quick",
        help="Config name or path (default: quick-test). Available: " + 
             ", ".join(CONFIGS.keys())
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List all available configurations"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without running"
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Override number of samples per dataset"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        help="Override datasets (comma-separated: cifar10,cifar100)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Override output directory"
    )
    
    args = parser.parse_args()
    
    # List available configs
    if args.list_configs:
        print("\n" + "="*80)
        print("AVAILABLE CONFIGURATIONS")
        print("="*80)
        for name, cfg in CONFIGS.items():
            datasets = ", ".join(cfg.datasets)
            samples = sum(cfg.num_samples.values())
            print(f"\n{name:20} | Datasets: {datasets:30} | Samples: {samples:5}")
            print(f"{'':20} | Surrogate: {cfg.surrogate_model}, Target: {cfg.target_model}")
            print(f"{'':20} | Query budget: {cfg.query_budget}, Steps: {cfg.num_steps}")
        print("="*80 + "\n")
        return 0
    
    # Load configuration
    if args.config.lower() in CONFIGS:
        cfg = CONFIGS[args.config.lower()]
    elif Path(args.config).exists():
        cfg = load_config_from_file(args.config)
        print(f"✓ Loaded config from {args.config}")
    else:
        print(f"✗ Unknown config: {args.config}")
        print(f"  Available: {', '.join(CONFIGS.keys())}")
        print(f"  Or provide path to JSON file")
        return 1
    
    # Apply overrides
    if args.samples:
        for dataset in cfg.datasets:
            cfg.num_samples[dataset] = args.samples
    
    if args.datasets:
        cfg.datasets = [d.strip() for d in args.datasets.split(",")]
        cfg.num_samples = {d: cfg.num_samples.get(d, 50) for d in cfg.datasets}
    
    if args.output:
        cfg.output_dir = args.output
    
    # Print summary
    print_config_summary(cfg)
    
    if args.dry_run:
        print("✓ Dry run complete (no experiment executed)")
        return 0
    
    # Run experiment
    try:
        run_experiment(cfg)
        print("\n✓ Experiment completed successfully!")
        return 0
    except KeyboardInterrupt:
        print("\n✗ Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
