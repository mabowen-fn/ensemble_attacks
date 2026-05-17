#!/usr/bin/env python3
"""
Run a minimal smoke run of the ported ELPD modules and print results.
"""
import torch
from types import SimpleNamespace
from ensemble_attacks.baselines import run_mifgsm
from ensemble_attacks.elpd_attack import elpd_blend_attack
from ensemble_attacks.elpd_blender import ELPDBlender
from ensemble_attacks.query_estimator import QueryEstimator

# Dummy surrogate: returns a stable random gradient (flat)
class DummySurrogate:
    def gradient(self, x, y):
        g = torch.randn(x.numel(), device=x.device)
        return g / (g.abs().sum() + 1e-12)

# Dummy target: predict() and a query function that returns random losses
class DummyTarget:
    def __init__(self):
        self.queries_used = 0
    def predict(self, x):
        # always return same label (0) for smoke run
        return 0
    def get_query_fn(self):
        def qfn(x_batch):
            B = x_batch.shape[0]
            self.queries_used += B
            # return random "losses" in [0,1]
            return torch.rand(B, device=x_batch.device)
        return qfn


def main():
    # Minimal config
    cfg = SimpleNamespace()
    cfg.attack = SimpleNamespace(num_steps=3, step_size=1e-2, epsilon=8/255,
                                 clip_min=0.0, clip_max=1.0, early_stop=False,
                                 query_budget=100)
    cfg.momentum = SimpleNamespace(mu=0.9, enabled=True)
    cfg.baselines = SimpleNamespace(run_static_hybrid=SimpleNamespace(eta_static=0.5))

    # Estimator and blender configs
    est_cfg = SimpleNamespace(method='nes', sigma=1e-2, n_samples=4, rao_blackwell=True)
    blender_cfg = SimpleNamespace(
        method='waic',
        waic=SimpleNamespace(min_samples_for_waic=4, llik_temperature=1.0),
        loo_psis=SimpleNamespace(pareto_k_threshold=0.7),
        cosine_var=SimpleNamespace(var_weight=0.1),
        eta_grid=SimpleNamespace(low=0.0, high=1.0, n_points=11),
        eta_min=0.0, eta_max=1.0, eta_ema_alpha=0.5
    )

    device = torch.device('cpu')
    x = torch.rand(3, 32, 32, device=device)
    label = 0
    surrogate = DummySurrogate()
    target = DummyTarget()

    print('Running MI-FGSM baseline (transfer only, no queries)')
    res = run_mifgsm(x, label, surrogate, target, cfg)
    print('MI-FGSM result: success=', res.success, 'steps=', res.steps_taken)

    print('\nRunning ELPD-blend attack (estimator + blender)')
    blender = ELPDBlender(blender_cfg, sigma=est_cfg.sigma)
    estimator = QueryEstimator(est_cfg, target.get_query_fn())
    res2 = elpd_blend_attack(x, label, surrogate, target, blender, estimator, cfg)
    print('ELPD-blend result: success=', res2.success, 'queries=', res2.queries_used)

if __name__ == '__main__':
    main()
