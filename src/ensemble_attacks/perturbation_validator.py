"""
Perturbation validation and constraint checking.

Critical for verifying that ensemble attacks maintain L∞ norm bounds.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List


def get_linf_perturbation(x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """
    Compute L∞ perturbation (maximum absolute change per sample).
    
    Args:
        x: Original images [batch, channels, height, width]
        x_adv: Adversarial examples [batch, channels, height, width]
        
    Returns:
        L∞ perturbations [batch] - max absolute change per sample
    """
    delta = (x_adv - x).abs()
    # Reshape to [batch, -1], then take max per sample
    linf = delta.view(delta.size(0), -1).max(dim=1).values
    return linf


def get_l2_perturbation(x: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 perturbation (Euclidean distance per sample).
    
    Args:
        x: Original images [batch, channels, height, width]
        x_adv: Adversarial examples [batch, channels, height, width]
        
    Returns:
        L2 perturbations [batch]
    """
    delta = x_adv - x
    l2 = delta.view(delta.size(0), -1).norm(p=2, dim=1)
    return l2


def check_linf_constraint(x: torch.Tensor, x_adv: torch.Tensor, epsilon: float, 
                         tolerance: float = 1e-6) -> Tuple[bool, torch.Tensor]:
    """
    Check if adversarial examples satisfy L∞ norm constraint.
    
    Args:
        x: Original images
        x_adv: Adversarial examples
        epsilon: Perturbation budget
        tolerance: Tolerance for numerical errors (default 1e-6)
        
    Returns:
        (satisfied, violations) where:
        - satisfied (bool): All samples satisfy constraint
        - violations (Tensor): Per-sample violations (0 if satisfied, else amount over)
    """
    linf = get_linf_perturbation(x, x_adv)
    violations = torch.clamp(linf - epsilon - tolerance, min=0)
    satisfied = (violations == 0).all().item()
    
    return satisfied, violations


def check_pixel_range(x_adv: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0,
                     tolerance: float = 1e-6) -> Tuple[bool, Dict[str, torch.Tensor]]:
    """
    Check if adversarial examples stay within pixel value range.
    
    Args:
        x_adv: Adversarial examples
        min_val: Minimum pixel value (default 0.0)
        max_val: Maximum pixel value (default 1.0)
        tolerance: Tolerance for numerical errors
        
    Returns:
        (satisfied, violations) where violations has 'below_min' and 'above_max'
    """
    violations = {
        'below_min': (x_adv.min(dim=1).values - min_val),
        'above_max': (x_adv.max(dim=1).values - max_val)
    }
    
    below_violations = torch.clamp(-violations['below_min'] - tolerance, min=0)
    above_violations = torch.clamp(violations['above_max'] - tolerance, min=0)
    
    satisfied = (below_violations == 0).all().item() and (above_violations == 0).all().item()
    
    return satisfied, {
        'below_min': below_violations,
        'above_max': above_violations
    }


def get_perturbation_statistics(x: torch.Tensor, x_adv: torch.Tensor, 
                               epsilon: float | None = None) -> Dict[str, float]:
    """
    Compute comprehensive perturbation statistics.
    
    Args:
        x: Original images
        x_adv: Adversarial examples
        epsilon: Perturbation budget (optional, for comparison)
        
    Returns:
        Dictionary with statistics
    """
    linf = get_linf_perturbation(x, x_adv)
    l2 = get_l2_perturbation(x, x_adv)
    
    stats = {
        'linf_mean': linf.mean().item(),
        'linf_median': linf.median().item(),
        'linf_std': linf.std().item(),
        'linf_min': linf.min().item(),
        'linf_max': linf.max().item(),
        'l2_mean': l2.mean().item(),
        'l2_median': l2.median().item(),
        'l2_std': l2.std().item(),
        'l2_min': l2.min().item(),
        'l2_max': l2.max().item(),
    }
    
    if epsilon is not None:
        stats['epsilon'] = epsilon
        stats['linf_vs_epsilon_ratio'] = (linf.mean() / epsilon).item()
        stats['max_linf_vs_epsilon_ratio'] = (linf.max() / epsilon).item()
        
        satisfied, violations = check_linf_constraint(x, x_adv, epsilon)
        stats['constraint_satisfied'] = satisfied
        if violations.sum() > 0:
            stats['constraint_violations_count'] = violations.gt(0).sum().item()
            stats['max_violation'] = violations.max().item()
        else:
            stats['constraint_violations_count'] = 0
            stats['max_violation'] = 0.0
    
    return stats


def print_perturbation_report(stats: Dict[str, float], attack_name: str = ""):
    """
    Pretty-print perturbation statistics.
    
    Args:
        stats: Statistics dictionary from get_perturbation_statistics()
        attack_name: Name of attack (optional)
    """
    prefix = f"[{attack_name}] " if attack_name else ""
    
    print(f"\n{prefix}PERTURBATION ANALYSIS")
    print("=" * 60)
    
    if 'epsilon' in stats:
        print(f"Epsilon Budget: {stats['epsilon']:.6f}")
        print(f"Constraint Satisfied: {'✓ YES' if stats['constraint_satisfied'] else '✗ NO'}")
        if stats['constraint_violations_count'] > 0:
            print(f"  Violations: {stats['constraint_violations_count']}")
            print(f"  Max violation: {stats['max_violation']:.6f}")
        print()
    
    print("L∞ Perturbation (max per-pixel change):")
    print(f"  Mean:   {stats['linf_mean']:.6f}")
    print(f"  Median: {stats['linf_median']:.6f}")
    print(f"  Std:    {stats['linf_std']:.6f}")
    print(f"  Min:    {stats['linf_min']:.6f}")
    print(f"  Max:    {stats['linf_max']:.6f}")
    
    print("\nL2 Perturbation (Euclidean distance):")
    print(f"  Mean:   {stats['l2_mean']:.6f}")
    print(f"  Median: {stats['l2_median']:.6f}")
    print(f"  Std:    {stats['l2_std']:.6f}")
    print(f"  Min:    {stats['l2_min']:.6f}")
    print(f"  Max:    {stats['l2_max']:.6f}")
    
    if 'linf_vs_epsilon_ratio' in stats:
        print(f"\nL∞ / ε Ratio:")
        print(f"  Mean: {stats['linf_vs_epsilon_ratio']:.4f} (should be ≤ 1.0)")
        print(f"  Max:  {stats['max_linf_vs_epsilon_ratio']:.4f} (should be ≤ 1.0)")
    
    print("=" * 60)


def project_to_lp_ball(delta: torch.Tensor, epsilon: float, p: float = float('inf'),
                      inplace: bool = False) -> torch.Tensor:
    """
    Project perturbation to Lp ball.
    
    Args:
        delta: Perturbations [batch, ...]
        epsilon: Norm bound
        p: Norm type (default: inf for L∞)
        inplace: Modify in place
        
    Returns:
        Projected perturbations
    """
    if not inplace:
        delta = delta.clone()
    
    if p == float('inf'):
        # L∞ projection: clamp to [-epsilon, epsilon]
        delta = torch.clamp(delta, min=-epsilon, max=epsilon)
    elif p == 2:
        # L2 projection: scale if norm exceeds epsilon
        batch_size = delta.size(0)
        delta_flat = delta.view(batch_size, -1)
        norms = torch.norm(delta_flat, p=2, dim=1, keepdim=True)
        scale = torch.clamp(norms, max=epsilon) / (norms + 1e-12)
        delta = (delta_flat * scale).view(delta.shape)
    else:
        raise NotImplementedError(f"Projection for p={p} not implemented")
    
    return delta


def reproject_to_epsilon_ball(x: torch.Tensor, x_adv: torch.Tensor, epsilon: float,
                             p: float = float('inf'), inplace: bool = False) -> torch.Tensor:
    """
    Re-project adversarial examples to ε-ball around original.
    
    CRITICAL: Fixes ensemble averaging constraint violation!
    After averaging, final result may violate ||x_adv - x||_p ≤ ε
    This function ensures constraint is satisfied.
    
    Args:
        x: Original images [batch, channels, height, width]
        x_adv: Adversarial examples (possibly unconstrained)
        epsilon: Perturbation budget
        p: Norm type (default: inf for L∞)
        inplace: Modify x_adv in place
        
    Returns:
        Re-projected adversarial examples guaranteed in ε-ball
    """
    if not inplace:
        x_adv = x_adv.clone()
    
    # Compute perturbation
    delta = x_adv - x
    
    # Project to Lp ball
    delta_projected = project_to_lp_ball(delta, epsilon, p=p, inplace=False)
    
    # Reconstruct and clamp to pixel range
    x_adv_projected = x + delta_projected
    x_adv_projected = torch.clamp(x_adv_projected, 0, 1)
    
    return x_adv_projected




def measure_linf_perturbation(x_adv: torch.Tensor, x: torch.Tensor) -> float:
    """
    Measure L∞ perturbation between adversarial and original image.
    
    Args:
        x_adv: Adversarial example (C, H, W) or (1, C, H, W)
        x: Original image (C, H, W) or (1, C, H, W)
        
    Returns:
        float: L∞ perturbation magnitude
    """
    # Handle both single images and batches
    if x_adv.dim() == 3:
        x_adv = x_adv.unsqueeze(0)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    
    linf_perturbs = get_linf_perturbation(x, x_adv)
    return linf_perturbs[0].item()


__all__ = [
    "get_linf_perturbation",
    "get_l2_perturbation", 
    "check_linf_constraint",
    "check_pixel_range",
    "get_perturbation_statistics",
    "print_perturbation_report",
    "project_to_lp_ball",
    "reproject_to_epsilon_ball",
    "measure_linf_perturbation",
]
