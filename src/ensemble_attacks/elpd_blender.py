"""
elpd_blender.py — Adaptive ELPD-Blend: the Bayesian core of the attack.

═══════════════════════════════════════════════════════════════════════════════
MATHEMATICAL FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════

Original biostatistics problem (Gower et al., 2019):
    We have two data sources:
        D_E  — small, unbiased RCT data   → our TARGET model queries
        D_O  — large, biased RWD          → our SURROGATE model gradients

    The power-likelihood model raises p(D_O | θ) to the power η ∈ [0,1]:
        p(θ | D_E, D_O) ∝ p(D_E | θ) · p(D_O | θ)^η · p(θ)

    η = 0 → ignore surrogate entirely (pure query attack)
    η = 1 → trust surrogate fully     (pure transfer attack)

    Optimal η is chosen by maximising the ELPD of the blended model
    evaluated on the held-out unbiased data D_E.

Our adversarial translation:
    "data point"      → a single gradient estimate  g ∈ ℝ^D
    "posterior draws" → the n_samples NES/SPSA gradient estimates {g_1,...,g_q}
                        from the TARGET model at the current iterate x_t
    "likelihood"      → a Gaussian centred on the blended mean gradient
                         L(g_target | g_blend, Σ) where Σ = σ²I (diagonal)
    "ELPD"            → expected log predictive density of the TARGET gradient
                        under the blended predictive distribution

Blended gradient:
    g_blend(η) = η · g_surrogate  +  (1 − η) · ĝ_target
    where ĝ_target = mean of the q NES draws.

ELPD approximations implemented:
    1. WAIC  (Watanabe–Akaike Information Criterion)
       elpd_WAIC = Σ_i log p̄(g_i | g_blend) − p_WAIC
       p_WAIC    = Σ_i Var_i[ log p(g_i | g_blend) ]      (penalty term)

       Derivation of p_WAIC in the gradient context:
         Let g_i  be the i-th NES draw  (our "observation").
         Under a Gaussian likelihood with mean μ = g_blend(η) and variance σ²:
           log p(g_i | μ, σ²) = −D/2 · log(2πσ²) − ||g_i − μ||² / (2σ²)
         Only the residual term varies with i, so:
           Var_i[log p(g_i | μ, σ²)] = Var_i[ ||g_i − μ||² ] / (2σ²)²
         We accumulate this variance across all D dimensions.

         Crucially: the penalty captures how inconsistent the NES draws are
         with the blended gradient — high variance → the blended mean is a
         poor predictor → low ELPD → η is penalised.

    2. LOO-PSIS  (Pareto-Smoothed Importance Sampling Leave-One-Out)
       LOO-PSIS gives a lower-variance estimate of ELPD than WAIC when the
       Pareto tail index k̂ < 0.7.  For k̂ ≥ 0.7 the IS weights are too
       heavy-tailed and we fall back to WAIC.

       IS weight for leaving out observation i:
         w_i(η) ∝ 1 / p(g_i | g_blend(η))   (ratio of full-data to LOO likelihood)
       In the Gaussian case this simplifies to:
         log w_i = +||g_i − μ(η)||² / (2σ²)   (remove observation i's contribution)
       These log weights are then Pareto-smoothed following Vehtari et al. (2017).

    3. Cosine-Variance fallback (fast, used when n_samples < min_samples_for_waic)
       elpd_proxy(η) = cosine_sim(g_blend(η), ĝ_target) − λ · Var[||g_i − g_blend||]
       This is NOT a proper ELPD estimator but is a fast, interpretable proxy
       that preserves the sign of ∂ELPD/∂η in most practical cases.

BvM caveat:
    The Bernstein-von Mises theorem guarantees posterior concentration around the
    MLE in smooth, well-identified models.  Neural loss landscapes violate this:
    they are non-convex, high-dimensional, and locally flat near saddle points.
    Our compensation: we operate on projected, normalised gradients (unit ℓ∞ ball)
    before computing ELPD, which reduces effective dimensionality and stabilises
    the variance estimates.  Additionally, the EMA smoothing of η prevents the
    blender from over-reacting to a single noisy ELPD estimate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class BlendResult:
    """Everything the caller needs from one ELPD optimisation step."""
    eta_raw:     float          # η chosen by ELPD maximisation (before EMA)
    eta_smoothed: float         # η after EMA smoothing
    g_blended:   Tensor         # final blended gradient, same shape as g_surrogate
    elpd_values: Tensor         # ELPD scores for each grid point (shape: [n_grid])
    eta_grid:    Tensor         # the η values that were evaluated  (shape: [n_grid])
    method_used: str            # which ELPD estimator was actually applied
    pareto_k:    float | None   # k̂ from PSIS (None when method != loo_psis)
    diagnostics: dict           # extra tensors for logging / W&B


# ── PSIS utilities ───────────────────────────────────────────────────────────

def _fit_pareto_tail(log_weights: Tensor, tail_fraction: float = 0.2) -> float:
    """
    Estimate the Pareto shape parameter k̂ for the upper tail of IS weights.

    Following Vehtari, Gelman & Gabry (2017), we fit a Generalised Pareto
    Distribution to the top (tail_fraction * n) log-weights using the
    Zhang & Stephens (2009) moment-matching estimator.

    A high k̂ (> 0.7) signals that a few draws dominate the IS sum, meaning
    LOO-PSIS is unreliable and we should fall back to WAIC.
    """
    n = log_weights.numel()
    m = max(int(math.ceil(tail_fraction * n)), 3)
    # Work with the top-m weights (sorted ascending, take last m)
    top_w = log_weights.sort().values[-m:]
    # Shift so minimum of tail == 0  (GPD location parameter)
    z = top_w - top_w[0]
    z_mean = z.mean().item()
    if z_mean == 0.0:
        return 0.0
    # Zhang-Stephens moment estimator: k̂ = mean(log(z/z_mean)) — fast O(m) estimate
    k_hat = z[z > 0].log().mean().item() - math.log(z_mean)
    return float(k_hat)


def _psis_smooth(log_weights: Tensor, tail_fraction: float = 0.2) -> Tensor:
    """
    Apply Pareto smoothing to raw log importance weights.

    We fit a GPD to the upper tail and replace the raw tail values with
    order-statistic expectations from the fitted GPD.  This reduces variance
    without introducing bias when k̂ < 0.7.

    Returns smoothed log weights (same shape as input).
    """
    n = log_weights.numel()
    m = max(int(math.ceil(tail_fraction * n)), 3)

    # Sort indices; we'll only smooth the top-m
    sorted_idx = log_weights.argsort()
    tail_idx = sorted_idx[-m:]
    tail_vals = log_weights[tail_idx]
    z = tail_vals - tail_vals.min()

    z_mean = z.mean().item()
    if z_mean == 0.0:
        return log_weights.clone()

    k_hat = z[z > 0].log().mean().item() - math.log(z_mean)

    if k_hat <= 0:
        # Exponential tail — no smoothing needed
        return log_weights.clone()

    # GPD quantile function: Q(p) = σ/k * ((1-p)^{-k} - 1), σ = k * z_mean
    # Expected order statistics: E[Z_(i)] ≈ Q((i - 0.5) / m)
    probs = (torch.arange(1, m + 1, device=log_weights.device).float() - 0.5) / m
    sigma = k_hat * z_mean
    expected_z = (sigma / k_hat) * ((1.0 - probs).pow(-k_hat) - 1.0)
    # Clamp to avoid numerical blow-up for k > 1 (extremely heavy tails)
    expected_z = expected_z.clamp(max=z.max().item() * 10)

    # Place smoothed values back at the tail positions
    smoothed = log_weights.clone()
    rank_in_tail = tail_vals.argsort()          # argsort within the tail
    smoothed[tail_idx[rank_in_tail]] = tail_vals.min() + expected_z

    return smoothed


# ── Gaussian log-likelihood helpers ──────────────────────────────────────────

def _gaussian_log_llik(
    observations: Tensor,   # (q, D)
    mean: Tensor,           # (D,)
    sigma: float,
) -> Tensor:
    """
    Element-wise Gaussian log-likelihood for each of the q observations.

    log p(g_i | μ, σ²) = −D/2·log(2πσ²) − ||g_i − μ||² / (2σ²)

    We return the per-observation scalar (summed over D), shape (q,).
    The constant −D/2·log(2πσ²) is shared across η values and cancels
    when comparing ELPD scores, so we include it for correctness but
    it doesn't affect η selection.
    """
    D = mean.numel()
    residuals = observations - mean.unsqueeze(0)          # (q, D)
    sq_norms  = residuals.pow(2).sum(dim=-1)              # (q,)
    log_const = -0.5 * D * math.log(2.0 * math.pi * sigma ** 2)
    return log_const - sq_norms / (2.0 * sigma ** 2)


# ── Core ELPDBlender class ────────────────────────────────────────────────────

class ELPDBlender:
    """
    Computes the optimal power-likelihood blending weight η at each attack step.

    Usage:
        blender = ELPDBlender(cfg.elpd_blender, cfg.query_estimator.sigma)
        result  = blender.step(g_surrogate, g_target_draws)
        # use result.g_blended for the next attack update
    """

    def __init__(self, blender_cfg, sigma: float) -> None:
        """
        Args:
            blender_cfg:  the elpd_blender sub-namespace from config.yaml
            sigma:        the NES/FD smoothing radius σ — doubles as the
                          Gaussian likelihood scale for WAIC/PSIS computations
        """
        self.cfg   = blender_cfg
        self.sigma = sigma

        # Build the η grid once (reused every step — no per-step allocation)
        self.eta_grid: Tensor = torch.linspace(
            blender_cfg.eta_grid.low,
            blender_cfg.eta_grid.high,
            blender_cfg.eta_grid.n_points,
        )                                       # shape: (n_grid,)

        # EMA state: initialise at 0.5 (equal weight until first observation)
        self._eta_ema: float = 0.5
        self._step_count: int = 0

    # ── public entry point ────────────────────────────────────────────────────

    def step(
        self,
        g_surrogate: Tensor,    # (D,)  — white-box surrogate gradient (flat)
        g_target_draws: Tensor, # (q, D) — q NES/SPSA draws from the target
    ) -> BlendResult:
        """
        Main per-step call.  Selects the optimal η and returns the blended gradient.

        Args:
            g_surrogate:    Gradient of the loss w.r.t. x computed from the
                            surrogate model.  Shape (D,) where D = C·H·W.
            g_target_draws: A stack of q independent gradient estimates from the
                            target model via NES/SPSA finite differences.
                            Shape (q, D).

        Returns:
            BlendResult — contains the blended gradient and full diagnostics.
        """
        assert g_surrogate.dim() == 1,       "g_surrogate must be flat (D,)"
        assert g_target_draws.dim() == 2,    "g_target_draws must be (q, D)"
        assert g_surrogate.shape[0] == g_target_draws.shape[1], \
            "Dimension mismatch between surrogate and target gradients"

        device = g_surrogate.device
        q, D   = g_target_draws.shape

        # ── Step 1: Project onto unit ℓ∞ ball for numerical stability ─────
        # BvM compensation: working in normalised gradient space reduces the
        # effective dimensionality from D ≈ 150k to a bounded [-1,1]^D space,
        # which stabilises variance estimates for the WAIC penalty term.
        g_sur_norm = self._linf_normalise(g_surrogate)           # (D,)
        g_tgt_norm = self._linf_normalise(g_target_draws)        # (q, D)

        ĝ_target = g_tgt_norm.mean(dim=0)                         # (q, D) → (D,)

        # Move η grid to device (lazy, one-time cost)
        if self.eta_grid.device != device:
            self.eta_grid = self.eta_grid.to(device)

        # ── Step 2: Choose ELPD method ────────────────────────────────────
        method = self.cfg.method
        if method == "waic" or (
            method == "loo_psis"
            and q < self.cfg.waic.min_samples_for_waic
        ):
            if q < self.cfg.waic.min_samples_for_waic:
                logger.debug(
                    "Only %d draws — falling back to cosine_var ELPD proxy.", q
                )
                method = "cosine_var"
            else:
                method = "waic"

        # ── Step 3: Compute ELPD scores over the η grid ───────────────────
        pareto_k = None

        if method == "waic":
            elpd_scores = self._waic_elpd(g_sur_norm, g_tgt_norm, ĝ_target)
        elif method == "loo_psis":
            elpd_scores, pareto_k = self._loo_psis_elpd(
                g_sur_norm, g_tgt_norm, ĝ_target
            )
            if pareto_k >= self.cfg.loo_psis.pareto_k_threshold:
                logger.warning(
                    "PSIS k̂ = %.3f ≥ %.1f — LOO-PSIS unreliable, falling back to WAIC.",
                    pareto_k, self.cfg.loo_psis.pareto_k_threshold,
                )
                elpd_scores = self._waic_elpd(g_sur_norm, g_tgt_norm, ĝ_target)
                method = "waic (fallback from psis)"
        else:  # cosine_var
            elpd_scores = self._cosine_var_elpd(g_sur_norm, g_tgt_norm, ĝ_target)

        # ── Step 4: Select η* = argmax ELPD ──────────────────────────────
        best_idx = elpd_scores.argmax().item()
        eta_raw  = float(self.eta_grid[best_idx].item())
        eta_raw  = max(self.cfg.eta_min, min(self.cfg.eta_max, eta_raw))

        cos_sur_tgt = float(
            torch.nn.functional.cosine_similarity(
                g_sur_norm.unsqueeze(0), ĝ_target.unsqueeze(0)
            ).item()
        )

        # ── Step 5: EMA smoothing to prevent erratic η trajectories ───────
        # Without smoothing, noisy ELPD estimates cause η to oscillate wildly,
        # wasting queries by repeatedly reversing the blend direction.
        alpha = self.cfg.eta_ema_alpha
        if self._step_count == 0:
            self._eta_ema = eta_raw   # cold start: no history to average with
        else:
            self._eta_ema = alpha * eta_raw + (1.0 - alpha) * self._eta_ema

        eta_smoothed = float(self._eta_ema)
        self._step_count += 1

        # ── Step 6: Blend in original (un-normalised) gradient space ──────
        # We selected η in normalised space, but we apply it to the raw
        # gradients so the step size is consistent with the attack's ε budget.
        g_blended = eta_smoothed * g_surrogate + (1.0 - eta_smoothed) * g_target_draws.mean(dim=0)

        diagnostics = {
            "eta_raw":          eta_raw,
            "eta_smoothed":     eta_smoothed,
            "elpd_at_eta_raw":  float(elpd_scores[best_idx].item()),
            "elpd_at_eta0":     float(elpd_scores[0].item()),   # pure query
            "elpd_at_eta1":     float(elpd_scores[-1].item()),  # pure transfer
            "cosine_sim_sur_tgt": cos_sur_tgt,
            "target_grad_var":  float(g_tgt_norm.var(dim=0).mean().item()),
            "n_target_draws":   q,
            "step":             self._step_count,
        }

        return BlendResult(
            eta_raw=eta_raw,
            eta_smoothed=eta_smoothed,
            g_blended=g_blended,
            elpd_values=elpd_scores.cpu(),
            eta_grid=self.eta_grid.cpu(),
            method_used=method,
            pareto_k=pareto_k,
            diagnostics=diagnostics,
        )

    # ── ELPD estimators ───────────────────────────────────────────────────────

    def _waic_elpd(
        self,
        g_sur: Tensor,      # (D,)
        g_tgt: Tensor,      # (q, D)  — already ℓ∞-normalised
        g_mean: Tensor,     # (D,)    — mean of g_tgt rows
    ) -> Tensor:
        """
        LOO cosine ELPD over the η grid.  Returns shape (n_grid,).

        Why previous formulations failed
        ─────────────────────────────────
        1. Gaussian log-likelihood with σ: log_const dominates, surface flat.
        2. Negative MSE (−||g_i−μ(η)||²): always minimised at η=0 since g_mean
           is the empirical centroid — trivially prefers pure query.
        3. Improvement formula (baseline_MSE − MSE(η)): mean is exactly 0 at
           η=0 because r0_i = g_tgt_i − g_mean sums to zero by definition.

        Correct approach — LOO cosine cross-validation
        ───────────────────────────────────────────────
        For each held-out draw i, compute the LOO mean of the remaining q−1 draws,
        blend it with the surrogate at weight η, and measure cosine similarity
        with the held-out draw:

            μ_LOO_i(η) = η·g_sur + (1−η)·mean_{j≠i}[g_j]
            score_i(η) = cos(μ_LOO_i(η), g_i)
            ELPD(η)    = mean_i[score_i(η)]

        This works because:
        - When the surrogate aligns with the draws, blending it in raises the
          prediction quality for each held-out draw → higher ELPD at large η.
        - When the surrogate opposes the draws, it harms predictions → lower ELPD
          at large η → argmax at small η.
        - Unlike the centroid-subtracted improvement, the LOO mean ≠ g_tgt_i,
          so the score is non-trivially informative.

        Efficient vectorised implementation
        ─────────────────────────────────────
        LOO mean_{j≠i} = (q·g_mean − g_tgt_i) / (q − 1)
