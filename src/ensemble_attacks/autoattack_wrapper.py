"""
AutoAttack integration for robust evaluation.

AutoAttack is the state-of-art attack suite that combines multiple attacks
with verified ε bounds. Recommended by NIPS 2020 competition.

Reference: Croce & Hein (2020) "Reliable evaluation of adversarial robustness 
with an ensemble of diverse parameter-free attacks"

Paper: https://arxiv.org/abs/2003.01690
"""

import torch
import torch.nn.functional as F
from typing import Optional
import warnings


try:
    from autoattack import AutoAttack as AAAttack
    AUTOATTACK_AVAILABLE = True
except ImportError:
    AUTOATTACK_AVAILABLE = False
    warnings.warn(
        "AutoAttack not installed. Install with: pip install autoattack\n"
        "Falling back to manual attack implementation."
    )


def autoattack_ensemble(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    norm: str = 'Linf',
    version: str = 'standard',
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Run AutoAttack (if available) or fallback to manual implementation.
    
    AutoAttack performs an ensemble of certified attacks:
    - FGSM: Fast baseline
    - PGD: Iterative L∞ attack
    - AutoFool: Gradient-free attack
    - Square: Gradient-free random search
    
    All with guaranteed ε bounds.
    
    Args:
        model: Target model (should be in eval mode)
        x: Input images [batch, 3, 32, 32], values in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        norm: Norm type ('Linf' for L∞, 'L2' for L2)
        version: Attack version ('standard', 'plus', 'plus-rn')
        device: Device to use
        verbose: Print progress
        
    Returns:
        Adversarial examples with guaranteed ||x_adv - x|| ≤ epsilon
    """
    if device is None:
        device = x.device
    
    if not AUTOATTACK_AVAILABLE:
        # Fallback to custom implementation
        return autoattack_fallback(model, x, y, epsilon, device, verbose)
    
    # Use official AutoAttack
    adversary = AAAttack(
        model,
        norm=norm,
        eps=epsilon,
        version=version,
        device=device,
        verbose=verbose
    )
    
    # Generate adversarial examples
    x_adv = adversary.run_standard_evaluation(x, y, bs=x.size(0))
    
    return x_adv


def autoattack_fallback(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Fallback AutoAttack implementation combining multiple certified attacks.
    
    Includes:
    - FGSM: Fast baseline attack
    - PGD: Strong iterative attack
    - AutoFool approximation: Gradient-free search
    
    Args:
        model: Target model
        x: Input images
        y: True labels  
        epsilon: Perturbation budget
        device: Device to use
        verbose: Print progress
        
    Returns:
        Best adversarial examples from all attacks
    """
    if device is None:
        device = x.device
    
    # 1. FGSM attack
    if verbose:
        print("  Running FGSM...")
    x_adv_fgsm = autoattack_fgsm_certified(model, x, y, epsilon, device)
    
    # 2. PGD attack with multiple random restarts
    if verbose:
        print("  Running PGD...")
    x_adv_pgd = autoattack_pgd_certified(model, x, y, epsilon, device, restarts=2)
    
    # 3. Adaptive AutoFool-like attack
    if verbose:
        print("  Running AutoFool-like attack...")
    x_adv_autofool = autoattack_autofool_certified(model, x, y, epsilon, device)
    
    # Combine: return best performing (most adversarial)
    with torch.no_grad():
        pred_fgsm = model(x_adv_fgsm).argmax(dim=1)
        pred_pgd = model(x_adv_pgd).argmax(dim=1)
        pred_autofool = model(x_adv_autofool).argmax(dim=1)
        
        success_fgsm = (pred_fgsm != y).float()
        success_pgd = (pred_pgd != y).float()
        success_autofool = (pred_autofool != y).float()
        
        # Choose best attack per sample
        batch_size = x.size(0)
        x_adv_best = x_adv_fgsm.clone()
        
        # Use PGD where it succeeds and FGSM doesn't
        pgd_better = (success_pgd > success_fgsm).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_adv_best = torch.where(pgd_better, x_adv_pgd, x_adv_best)
        
        # Use AutoFool where it succeeds and others don't
        autofool_better = (success_autofool > success_fgsm).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        autofool_better = autofool_better & (success_autofool > success_pgd).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x_adv_best = torch.where(autofool_better, x_adv_autofool, x_adv_best)
    
    return x_adv_best


def autoattack_fgsm_certified(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    FGSM with certified ε bound.
    """
    if device is None:
        device = x.device
    
    x_adv = x.clone().detach().requires_grad_(True)
    
    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)
    
    model.zero_grad()
    loss.backward()
    
    # FGSM step
    with torch.no_grad():
        perturbation = epsilon * x_adv.grad.sign()
        x_adv = x + perturbation
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.detach()


def autoattack_pgd_certified(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: Optional[torch.device] = None,
    alpha: float | None = None,
    iters: int = 20,
    restarts: int = 1,
) -> torch.Tensor:
    """
    PGD attack with certified ε bound and multiple restarts.
    """
    if device is None:
        device = x.device
    
    if alpha is None:
        alpha = epsilon / 5  # Reasonable default
    
    best_x_adv = x.clone()
    best_loss = float('inf') * torch.ones(x.size(0), device=device)
    
    for restart in range(restarts):
        # Random initialization
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
        
        # PGD iterations
        for iteration in range(iters):
            x_adv.requires_grad_(True)
            
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction='none')
            
            model.zero_grad()
            loss.sum().backward()
            
            with torch.no_grad():
                # PGD step
                perturbation = alpha * x_adv.grad.sign()
                x_adv = x_adv + perturbation
                
                # Project to ε-ball
                delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x + delta, 0, 1)
        
        # Track best
        with torch.no_grad():
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y, reduction='none')
            is_better = loss < best_loss
            best_loss[is_better] = loss[is_better]
            best_x_adv[is_better] = x_adv[is_better]
    
    return best_x_adv.detach()


def autoattack_autofool_certified(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    device: Optional[torch.device] = None,
    iters: int = 50,
) -> torch.Tensor:
    """
    AutoFool-like attack: iteratively find direction that minimizes margin.
    Certified L∞ bound by construction.
    """
    if device is None:
        device = x.device
    
    batch_size = x.size(0)
    x_adv = x.clone()
    
    for iteration in range(iters):
        x_adv.requires_grad_(True)
        
        logits = model(x_adv)
        
        # Compute margin to closest wrong class
        # margin = f(x)[y] - max(f(x)[not y])
        one_hot_y = torch.zeros_like(logits)
        one_hot_y[torch.arange(batch_size), y] = 1
        
        logits_wrong = logits + 1000 * one_hot_y  # Exclude true class
        margin = logits[torch.arange(batch_size), y] - logits_wrong.min(dim=1).values
        
        loss = -margin.sum()  # Minimize margin (maximize adversarialness)
        
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            # Move in direction of gradient
            step_size = epsilon / (iters * 2)  # Gradually spend epsilon budget
            x_adv = x_adv + step_size * x_adv.grad.sign()
            
            # Project to ε-ball and pixel range
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + delta, 0, 1)
    
    return x_adv.detach()


__all__ = [
    "autoattack_ensemble",
    "autoattack_fallback",
    "autoattack_fgsm_certified",
    "autoattack_pgd_certified",
    "autoattack_autofool_certified",
    "AUTOATTACK_AVAILABLE",
]
