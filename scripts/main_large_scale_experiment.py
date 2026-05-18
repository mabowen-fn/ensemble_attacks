#!/usr/bin/env python3
"""
Large-Scale Power Likelihood Attacks Experiment.

This script evaluates ELPD-Blend vs baselines across multiple datasets and architectures
with rigorous statistical analysis and comprehensive logging.

Features:
- Multi-dataset support (CIFAR-10, CIFAR-100, TinyImageNet)
- Multiple model architectures (ResNet18, VGG16, DenseNet121)
- Baseline attacks (MI-FGSM, NES-Only, Static Hybrid)
- Statistical significance testing
- Production-quality error handling and logging
- MPS/CUDA/CPU device support with auto-tuning
- Reproducibility with fixed seeds
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy import stats
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ensemble_attacks.data import get_dataset_loaders
from ensemble_attacks.models import get_model
from ensemble_attacks.utils import get_device, set_random_seed, get_device_info
from ensemble_attacks.baselines import run_mifgsm, run_nes_only, run_static_hybrid
from ensemble_attacks.elpd_attack import elpd_blend_attack
from ensemble_attacks.elpd_blender import ELPDBlender
from ensemble_attacks.query_estimator import QueryEstimator
from ensemble_attacks.perturbation_validator import measure_linf_perturbation


logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    seed: int = 42
    datasets: List[str] = None
    surrogate_model: str = "resnet18"
    target_model: str = "vgg16"
    num_samples: Dict[str, int] = None  # samples per dataset

    # Attack parameters
    epsilon: float = 8.0 / 255.0
    step_size: float = 2.0 / 255.0
    num_steps: int = 25
    query_budget: int = 1000

    # ELPD parameters
    elpd_method: str = "waic"
    min_samples_for_waic: int = 6
    n_nes_samples: int = 8
    nes_sigma: float = 1.5e-2
    eta_ema_alpha: float = 0.4

    # Model loading
    use_pretrained: bool = True  # Use ImageNet pretrained weights
    surrogate_checkpoint: Optional[str] = None  # Path to surrogate checkpoint
    target_checkpoint: Optional[str] = None  # Path to target checkpoint

    # Device and I/O
    device_str: Optional[str] = None  # "cuda", "mps", "cpu", or None (auto)
    output_dir: str = "./outputs/large_scale_experiment"
    data_dir: str = "./data"
    batch_size: Optional[int] = None  # None = auto-tune
    num_workers: Optional[int] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = ["cifar10"]  # Start small, can add cifar100, tinyimagenet
        if self.num_samples is None:
            self.num_samples = {d: 50 for d in self.datasets}  # 50 samples per dataset for testing


class SurrogateModel:
    """White-box surrogate model wrapper."""
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device
    
    def gradient(self, x: torch.Tensor, true_label: int) -> torch.Tensor:
        """Compute gradient of CE loss w.r.t. x."""
        x_batch = x.unsqueeze(0).clone().detach().requires_grad_(True)
        logits = self.model(x_batch)
        loss = F.cross_entropy(logits, torch.tensor([true_label], device=self.device))
        self.model.zero_grad()
        loss.backward()
        grad = x_batch.grad.clone().detach().squeeze(0)
        return grad.view(-1)


class TargetModel:
    """Black-box target model wrapper with query counting."""
    def __init__(self, model, device, query_budget: int = 1000):
        self.model = model.eval()
        self.device = device
        self.query_budget = query_budget
        self.queries_used = 0
    
    def predict(self, x: torch.Tensor) -> int:
        """Single image prediction (C, H, W)."""
        self.queries_used += 1
        if self.queries_used > self.query_budget:
            raise RuntimeError(f"Query budget exceeded: {self.queries_used} > {self.query_budget}")
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))
            return logits.argmax(dim=1).item()
    
    def get_query_fn(self):
        """Batch query function for NES/SPSA."""
        def query_fn(x_batch: torch.Tensor) -> torch.Tensor:
            self.queries_used += x_batch.shape[0]
            if self.queries_used > self.query_budget:
                raise RuntimeError(f"Query budget exceeded: {self.queries_used} > {self.query_budget}")
            with torch.no_grad():
                logits = self.model(x_batch)
                log_probs = F.log_softmax(logits, dim=1)
                losses = -log_probs.max(dim=1).values
                return losses
        return query_fn
    
    def reset_queries(self):
        self.queries_used = 0


def get_config() -> SimpleNamespace:
    """Get detailed attack configuration."""
    cfg = SimpleNamespace()
    
    cfg.attack = SimpleNamespace(
        num_steps=25,
        step_size=2 / 255,
        epsilon=8 / 255,
        clip_min=0.0,
        clip_max=1.0,
        early_stop=True,
        query_budget=1000
    )
    
    cfg.momentum = SimpleNamespace(mu=0.9, enabled=True)
    
    cfg.query_estimator = SimpleNamespace(
        method='nes',
        sigma=1.5e-2,
        n_samples=8,
        rao_blackwell=True
    )
    
    cfg.elpd_blender = SimpleNamespace(
        method='waic',
        waic=SimpleNamespace(min_samples_for_waic=6, llik_temperature=0.5),
        loo_psis=SimpleNamespace(pareto_k_threshold=0.7),
        cosine_var=SimpleNamespace(var_weight=0.1),
        eta_grid=SimpleNamespace(low=0.0, high=1.0, n_points=21),
        eta_min=0.0,
        eta_max=1.0,
        eta_ema_alpha=0.4
    )
    
    cfg.baselines = SimpleNamespace(
        run_static_hybrid=SimpleNamespace(eta_static=0.5)
    )
    
    return cfg


def run_single_attack(
    x: torch.Tensor,
    true_label: int,
    attack_name: str,
    surrogate: SurrogateModel,
    target: TargetModel,
    cfg: SimpleNamespace,
    device: torch.device,
) -> Dict:
    """Run a single attack and return metrics."""
    target.reset_queries()
    
    try:
        if attack_name == "mifgsm":
            result = run_mifgsm(x, true_label, surrogate, target, cfg)
        elif attack_name == "nes_only":
            estimator = QueryEstimator(cfg.query_estimator, target.get_query_fn())
            result = run_nes_only(x, true_label, target, estimator, cfg)
        elif attack_name == "static_hybrid":
            estimator = QueryEstimator(cfg.query_estimator, target.get_query_fn())
            result = run_static_hybrid(x, true_label, surrogate, target, estimator, cfg)
        elif attack_name == "elpd_blend":
            blender = ELPDBlender(cfg.elpd_blender, cfg.query_estimator.sigma)
            estimator = QueryEstimator(cfg.query_estimator, target.get_query_fn())
            result = elpd_blend_attack(x, true_label, surrogate, target, blender, estimator, cfg)
        else:
            raise ValueError(f"Unknown attack: {attack_name}")
        
        linf = measure_linf_perturbation(result.adv_x, x)
        
        return {
            "success": result.success,
            "queries": result.queries_used,
            "steps": result.steps_taken,
            "linf": float(linf),
            "eta_history": result.eta_history if hasattr(result, 'eta_history') else [],
            "query_history": result.query_history if hasattr(result, 'query_history') else [],
        }
    except Exception as e:
        logger.warning(f"Attack {attack_name} failed: {e}")
        return {
            "success": False,
            "queries": target.queries_used,
            "steps": 0,
            "linf": float('nan'),
            "error": str(e),
        }


def run_experiment(cfg: ExperimentConfig) -> None:
    """Run the main large-scale experiment."""
    
    # Setup
    logger.info("=" * 80)
    logger.info("POWER LIKELIHOOD ATTACKS: LARGE-SCALE EXPERIMENT")
    logger.info("=" * 80)
    
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if cfg.device_str:
        device = torch.device(cfg.device_str)
    else:
        device = get_device(prefer_cuda=False)  # Auto-select
    
    logger.info(f"Device: {device}")
    logger.info(f"Device info: {json.dumps(get_device_info(), indent=2, default=str)}")
    
    set_random_seed(cfg.seed, deterministic=False)
    
    # Get detailed config
    attack_cfg = get_config()
    attack_cfg.attack.query_budget = cfg.query_budget
    attack_cfg.attack.epsilon = cfg.epsilon
    attack_cfg.attack.step_size = cfg.step_size
    attack_cfg.attack.num_steps = cfg.num_steps
    attack_cfg.query_estimator.sigma = cfg.nes_sigma
    attack_cfg.query_estimator.n_samples = cfg.n_nes_samples
    
    # Store all results
    all_results = []
    
    # Iterate over datasets
    for dataset in cfg.datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"DATASET: {dataset.upper()}")
        logger.info(f"{'='*80}")
        
        try:
            # Load dataset
            logger.info(f"Loading {dataset} dataset...")
            _, test_loader, num_classes = get_dataset_loaders(
                dataset=dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                device=device,
                data_dir=cfg.data_dir,
            )
            
            num_samples = cfg.num_samples.get(dataset, 10)
            logger.info(f"Running on {num_samples} test samples (num_classes={num_classes})")
            
            # Load models
            surrogate_model = get_model(
                cfg.surrogate_model,
                num_classes=num_classes,
                pretrained=cfg.use_pretrained,
                checkpoint_path=cfg.surrogate_checkpoint
            ).to(device)
            target_model = get_model(
                cfg.target_model,
                num_classes=num_classes,
                pretrained=cfg.use_pretrained,
                checkpoint_path=cfg.target_checkpoint
            ).to(device)

            surrogate = SurrogateModel(surrogate_model, device)
            target = TargetModel(target_model, device, query_budget=cfg.query_budget)

            logger.info(f"Surrogate: {cfg.surrogate_model} (pretrained={cfg.use_pretrained})")
            logger.info(f"Target: {cfg.target_model} (pretrained={cfg.use_pretrained})")
            if cfg.surrogate_checkpoint:
                logger.info(f"  Surrogate checkpoint: {cfg.surrogate_checkpoint}")
            if cfg.target_checkpoint:
                logger.info(f"  Target checkpoint: {cfg.target_checkpoint}")
            
            # Attack names
            attacks = ["mifgsm", "nes_only", "static_hybrid", "elpd_blend"]
            
            # Run attacks on samples
            dataset_results = {attack: [] for attack in attacks}
            
            sample_count = 0
            with tqdm(test_loader, desc="Samples", unit="sample") as pbar:
                for x_batch, y_batch in pbar:
                    if sample_count >= num_samples:
                        break
                    
                    x = x_batch[0].to(device, non_blocking=True, dtype=torch.float32)
                    true_label = y_batch[0].item()
                    
                    for attack_name in attacks:
                        result = run_single_attack(
                            x, true_label, attack_name, surrogate, target, attack_cfg, device
                        )
                        dataset_results[attack_name].append(result)
                        pbar.set_postfix({f"{attack_name[:3]}": "✓" if result.get("success") else "✗"})
                    
                    sample_count += 1
            
            # Aggregate results for this dataset
            logger.info(f"\nResults for {dataset}:")
            logger.info("-" * 80)
            
            for attack_name in attacks:
                results = dataset_results[attack_name]
                successes = sum(1 for r in results if r.get("success"))
                asr = successes / len(results) if results else 0.0
                avg_queries = np.mean([r.get("queries", 0) for r in results]) if results else 0
                avg_linf = np.mean([r.get("linf", 0) for r in results if not np.isnan(r.get("linf", 0))])
                
                logger.info(f"\n{attack_name.upper()}")
                logger.info(f"  ASR: {asr:.1%}")
                logger.info(f"  Avg Queries: {avg_queries:.1f} / {cfg.query_budget}")
                logger.info(f"  Avg L∞: {avg_linf:.6f} (ε = {cfg.epsilon:.6f})")
                
                # Store individual results
                for idx, result in enumerate(results):
                    all_results.append({
                        "dataset": dataset,
                        "surrogate": cfg.surrogate_model,
                        "target": cfg.target_model,
                        "attack": attack_name,
                        "sample_id": idx,
                        "success": result.get("success", False),
                        "queries": result.get("queries", 0),
                        "steps": result.get("steps", 0),
                        "linf": result.get("linf", np.nan),
                        "error": result.get("error", ""),
                    })
        
        except Exception as e:
            logger.error(f"Error processing dataset {dataset}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    # Export results
    logger.info(f"\n{'='*80}")
    logger.info("EXPORTING RESULTS")
    logger.info(f"{'='*80}")
    
    export_results(all_results, cfg.output_dir)


def export_results(results: List[Dict], output_dir: str) -> None:
    """Export results to CSV and generate summary statistics."""
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        logger.warning("No results to export")
        return
    
    # Save raw results
    csv_path = Path(output_dir) / "results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results to {csv_path}")
    
    # Generate summary by dataset and attack
    logger.info("\n" + "="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)
    
    summary = df.groupby(["dataset", "attack"]).agg({
        "success": ["sum", "count", "mean"],
        "queries": ["mean", "std"],
        "linf": ["mean", "std"],
    }).round(4)
    
    logger.info(f"\n{summary}")
    summary.to_csv(Path(output_dir) / "summary.csv")
    
    # Statistical comparison
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL COMPARISON (ELPD-BLEND vs BASELINES)")
    logger.info("="*80)
    
    for dataset in df["dataset"].unique():
        logger.info(f"\nDataset: {dataset}")
        dataset_df = df[df["dataset"] == dataset]
        
        elpd_asr = dataset_df[dataset_df["attack"] == "elpd_blend"]["success"].values
        
        for baseline in ["mifgsm", "nes_only", "static_hybrid"]:
            baseline_asr = dataset_df[dataset_df["attack"] == baseline]["success"].values
            
            if len(elpd_asr) > 0 and len(baseline_asr) > 0:
                # T-test
                t_stat, p_value = stats.ttest_ind(elpd_asr, baseline_asr)
                logger.info(f"  ELPD vs {baseline}: p={p_value:.4f}, t={t_stat:.4f}")
    
    logger.info(f"\nAll results saved to {output_dir}")


def main():
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create config
    cfg = ExperimentConfig(
        datasets=["cifar10"],  # Start with CIFAR-10
        surrogate_model="resnet18",
        target_model="vgg16",
        num_samples={"cifar10": 50},  # 50 samples for testing, increase for full run
        num_steps=25,
        query_budget=1000,
        epsilon=8.0 / 255.0,
        step_size=2.0 / 255.0,
    )
    
    try:
        run_experiment(cfg)
        logger.info("\n✓ Experiment completed successfully!")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
