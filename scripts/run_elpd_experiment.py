#!/usr/bin/env python3
"""
Experiment: Compare ELPD-Blend attack against baselines (MI-FGSM, NES-only, Static-Hybrid)
on CIFAR-10 with a surrogate and target model.

This tests the novel black-and-white-power-likelihood technique for adaptive blending.
"""

import os
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from tqdm import tqdm
import json
from pathlib import Path

from ensemble_attacks.models import get_resnet18
from ensemble_attacks.data import get_cifar10_loaders
from ensemble_attacks.utils import get_device
from ensemble_attacks.baselines import run_mifgsm, run_nes_only, run_static_hybrid
from ensemble_attacks.elpd_attack import elpd_blend_attack
from ensemble_attacks.elpd_blender import ELPDBlender
from ensemble_attacks.query_estimator import QueryEstimator


# ── Model wrappers for attack compatibility ──────────────────────────────────

class SurrogateModel:
    """White-box surrogate model: compute gradients."""
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def gradient(self, x, true_label):
        """Compute gradient of cross-entropy loss w.r.t. x (flat)."""
        # x is (C, H, W), add batch dim for model
        x_batch = x.unsqueeze(0).clone().detach().requires_grad_(True)
        logits = self.model(x_batch)
        loss = F.cross_entropy(logits, torch.tensor([true_label], device=self.device))
        self.model.zero_grad()
        loss.backward()
        grad = x_batch.grad.clone().detach().squeeze(0)
        return grad.view(-1)  # Flatten


class TargetModel:
    """Black-box target model: only query interface."""
    def __init__(self, model, device, query_budget=1000):
        self.model = model
        self.device = device
        self.queries_used = 0
        self.query_budget = query_budget

    def predict(self, x):
        """Predict class for a single image x (C, H, W)."""
        self.queries_used += 1
        if self.queries_used > self.query_budget:
            raise RuntimeError(f"Query budget exceeded: {self.queries_used} > {self.query_budget}")
        with torch.no_grad():
            logits = self.model(x.unsqueeze(0))
            return logits.argmax(dim=1).item()

    def get_query_fn(self):
        """Return a batch query function for NES/SPSA."""
        def query_fn(x_batch):
            """x_batch: (B, C, H, W), returns loss (B,)."""
            self.queries_used += x_batch.shape[0]
            if self.queries_used > self.query_budget:
                raise RuntimeError(f"Query budget exceeded: {self.queries_used} > {self.query_budget}")
            with torch.no_grad():
                logits = self.model(x_batch)
                # Use negative log softmax probabilities as loss
                log_probs = F.log_softmax(logits, dim=1)
                losses = -log_probs.max(dim=1).values  # Negative max log prob
                return losses
        return query_fn

    def reset_queries(self):
        self.queries_used = 0


# ── Experiment config ────────────────────────────────────────────────────────

def get_experiment_config():
    cfg = SimpleNamespace()

    # Attack hyperparameters
    cfg.attack = SimpleNamespace(
        num_steps=20,
        step_size=2 / 255,
        epsilon=8 / 255,
        clip_min=0.0,
        clip_max=1.0,
        early_stop=True,
        query_budget=1000
    )

    # Momentum
    cfg.momentum = SimpleNamespace(
        mu=0.9,
        enabled=True
    )

    # Query estimator (NES)
    cfg.query_estimator = SimpleNamespace(
        method='nes',
        sigma=1e-2,
        n_samples=10,  # 10 pairs = 20 queries per step
        rao_blackwell=True
    )

    # ELPD Blender config
    cfg.elpd_blender = SimpleNamespace(
        method='waic',
        waic=SimpleNamespace(min_samples_for_waic=8, llik_temperature=1.0),
        loo_psis=SimpleNamespace(pareto_k_threshold=0.7),
        cosine_var=SimpleNamespace(var_weight=0.1),
        eta_grid=SimpleNamespace(low=0.0, high=1.0, n_points=21),
        eta_min=0.0,
        eta_max=1.0,
        eta_ema_alpha=0.3
    )

    # Baselines
    cfg.baselines = SimpleNamespace(
        run_static_hybrid=SimpleNamespace(eta_static=0.5)
    )

    return cfg


# ── Main experiment ──────────────────────────────────────────────────────────

def run_attack_on_sample(x, true_label, attack_name, surrogate, target, cfg, device):
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

        # Compute perturbation norm
        linf = (result.adv_x - x).abs().max().item()

        return {
            "success": result.success,
            "queries": result.queries_used,
            "steps": result.steps_taken,
            "linf": linf,
            "eta_history": result.eta_history if hasattr(result, 'eta_history') else [],
        }
    except Exception as e:
        return {
            "success": False,
            "queries": target.queries_used,
            "steps": 0,
            "linf": 0.0,
            "error": str(e)
        }


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path("outputs/elpd_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading CIFAR-10...")
    _, test_loader = get_cifar10_loaders(batch_size=1, num_workers=0)

    # Load or train surrogate model
    surrogate_path = "resnet18_cifar10.pt"
    if os.path.exists(surrogate_path):
        print(f"Loading surrogate from {surrogate_path}")
        surrogate_net = get_resnet18(num_classes=10)
        surrogate_net.load_state_dict(torch.load(surrogate_path, map_location=device))
    else:
        print("Surrogate model not found. Please run: python scripts/train_resnet18.py")
        return

    surrogate_net.to(device).eval()

    # Create target model (same architecture, same weights for now)
    target_net = get_resnet18(num_classes=10)
    target_net.load_state_dict(torch.load(surrogate_path, map_location=device))
    target_net.to(device).eval()

    # Wrap models
    surrogate = SurrogateModel(surrogate_net, device)
    target = TargetModel(target_net, device, query_budget=1000)

    # Experiment config
    cfg = get_experiment_config()

    # Attack names
    attacks = ["mifgsm", "nes_only", "static_hybrid", "elpd_blend"]

    # Run attacks on first N samples
    num_samples = 20
    all_results = {attack: [] for attack in attacks}

    print(f"\nRunning attacks on {num_samples} CIFAR-10 test samples...")
    for batch_idx, (x_batch, y_batch) in enumerate(test_loader):
        if batch_idx >= num_samples:
            break

        x = x_batch[0].to(device, dtype=torch.float32)
        true_label = y_batch[0].item()

        print(f"\n[{batch_idx+1}/{num_samples}] Label: {true_label}")

        for attack_name in attacks:
            print(f"  Running {attack_name}...", end=" ")
            result = run_attack_on_sample(x, true_label, attack_name, surrogate, target, cfg, device)
            all_results[attack_name].append(result)
            status = "✓" if result.get("success") else "✗"
            queries = result.get("queries", 0)
            print(f"{status} (queries: {queries})")

    # Aggregate results
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)

    summary = {}
    for attack_name in attacks:
        results = all_results[attack_name]
        successes = sum(1 for r in results if r.get("success"))
        asr = successes / len(results) if results else 0.0
        avg_queries = sum(r.get("queries", 0) for r in results) / len(results) if results else 0
        avg_linf = sum(r.get("linf", 0) for r in results) / len(results) if results else 0

        summary[attack_name] = {
            "asr": asr,
            "avg_queries": avg_queries,
            "avg_linf": avg_linf,
            "num_samples": len(results)
        }

        print(f"\n{attack_name.upper()}")
        print(f"  ASR (Attack Success Rate): {asr:.2%}")
        print(f"  Avg Queries: {avg_queries:.1f}")
        print(f"  Avg L∞ perturbation: {avg_linf:.4f} (ε = {cfg.attack.epsilon:.4f})")

    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": summary,
            "detailed": all_results,
            "config": {
                "num_samples": num_samples,
                "epsilon": float(cfg.attack.epsilon),
                "num_steps": cfg.attack.num_steps,
                "query_budget": cfg.attack.query_budget,
            }
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print hypothesis check
    print("\n" + "="*70)
    print("HYPOTHESIS CHECK: Black-and-White Power-Likelihood")
    print("="*70)
    elpd_asr = summary["elpd_blend"]["asr"]
    mifgsm_asr = summary["mifgsm"]["asr"]
    nes_asr = summary["nes_only"]["asr"]

    print(f"\nELPD-Blend ASR: {elpd_asr:.2%}")
    print(f"MI-FGSM ASR (transfer only): {mifgsm_asr:.2%}")
    print(f"NES-only ASR (query only): {nes_asr:.2%}")

    if elpd_asr > mifgsm_asr and elpd_asr > nes_asr:
        print("\n✓ HYPOTHESIS CONFIRMED: Adaptive blending outperforms both pure transfer and pure query!")
    else:
        print("\n✗ Hypothesis not confirmed yet. Consider tuning ELPD hyperparameters.")

    print(f"\nQuery efficiency (ELPD vs baselines):")
    print(f"  ELPD-Blend: {summary['elpd_blend']['avg_queries']:.1f} queries")
    print(f"  Static Hybrid: {summary['static_hybrid']['avg_queries']:.1f} queries")


if __name__ == "__main__":
    main()
