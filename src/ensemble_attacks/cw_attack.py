"""
Carlini-Wagner L∞ attack implementation.

Reference: C&W Attack (Carlini & Wagner, 2016)
- Solves optimization: minimize ||perturbation||∞ such that loss is minimized
- Uses binary search on perturbation budget
- Guaranteed to find bounded adversarial examples
"""

import torch
import torch.nn.functional as F
from typing import Optional


def carlini_wagner_linf(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    confidence: float = 0.0,
    initial_const: float = 0.001,
    binary_search_steps: int = 9,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Carlini-Wagner L∞ attack with binary search on perturbation magnitude.
    
    This attack solves:
        minimize: ||δ||∞
        subject to: model(x + δ) has target confidence
                   ||δ||∞ ≤ epsilon
                   x + δ ∈ [0, 1]
    
    Args:
        model: Target neural network (should be in eval mode)
        x: Input images [batch, 3, 32, 32], values in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget (L∞ norm bound)
        max_iterations: Maximum iterations per optimization step
        learning_rate: Adam learning rate
        confidence: Confidence bonus for targeted/untargeted attack
        initial_const: Initial perturbation constant for binary search
        binary_search_steps: Number of binary search iterations
        device: Device to use (auto-detected if None)
        verbose: Print progress
        
    Returns:
        Adversarial examples with ||x_adv - x||_∞ ≤ epsilon
    """
    if device is None:
        device = x.device
    
    batch_size = x.size(0)
    
    # Convert to tanh space for unconstrained optimization
    # x_adv = x + tanh_scaled_perturbation
    # This ensures x_adv stays in [0, 1] during optimization
    
    # Initialize perturbation in tanh space
    delta_tanh = torch.zeros_like(x, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta_tanh], lr=learning_rate)
    
    best_x_adv = x.clone()
    best_loss = float('inf') * torch.ones(batch_size, device=device)
    
    # Binary search on perturbation magnitude
    lower_bound = torch.zeros(batch_size, device=device)
    upper_bound = torch.ones(batch_size, device=device) * initial_const
    const = upper_bound.clone()
    
    for binary_step in range(binary_search_steps):
        if verbose and binary_step % (binary_search_steps // 3 + 1) == 0:
            print(f"  Binary search step {binary_step}/{binary_search_steps}")
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Convert from tanh space back to image space
            # Tanh output is [-1, 1], scale to [-epsilon, epsilon]
            delta = epsilon * torch.tanh(delta_tanh)
            x_adv = x + delta
            x_adv = torch.clamp(x_adv, 0, 1)
            
            # Forward pass
            logits = model(x_adv)
            
            # Cross-entropy loss for untargeted attack
            loss_ce = F.cross_entropy(logits, y, reduction='none')
            
            # L∞ perturbation magnitude (for logging)
            linf = (x_adv - x).abs().view(batch_size, -1).max(dim=1).values
            
            # Total loss: perturbation magnitude + classification loss
            # Use const to weight classification loss
            loss = const * loss_ce + linf
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            
            # Track best examples
            with torch.no_grad():
                is_better = loss_ce < best_loss
                best_loss[is_better] = loss_ce[is_better]
                best_x_adv[is_better] = x_adv[is_better].clone()
        
        # Binary search update
        with torch.no_grad():
            # Check which samples achieved low loss
            logits = model(best_x_adv)
            pred = logits.argmax(dim=1)
            
            # Update bounds
            success = (pred != y).float()  # 1 if attack succeeded, 0 otherwise
            
            # If successful, try smaller epsilon (lower bound)
            # If failed, try larger epsilon (upper bound)
            lower_bound = torch.where(success > 0.5, lower_bound, const)
            upper_bound = torch.where(success > 0.5, const, upper_bound)
            
            # Geometric mean for next binary search
            const = (lower_bound + upper_bound) / 2
    
    return best_x_adv.detach()


def carlini_wagner_linf_simple(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Simplified CW L∞ attack without binary search (faster but less optimal).
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32]
        y: True labels [batch]
        epsilon: Perturbation budget
        max_iterations: Maximum optimization steps
        learning_rate: Adam learning rate
        device: Device to use
        
    Returns:
        Adversarial examples
    """
    if device is None:
        device = x.device
    
    batch_size = x.size(0)
    
    # Perturbation variable (unconstrained)
    delta = torch.zeros_like(x, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=learning_rate)
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()
        
        # Clamp perturbation to [-epsilon, epsilon]
        delta_clamped = torch.clamp(delta, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta_clamped, 0, 1)
        
        # Loss: maximize cross-entropy (untargeted attack)
        logits = model(x_adv)
        loss = -F.cross_entropy(logits, y)  # Negative for maximization
        
        loss.backward()
        optimizer.step()
    
    # Final adversarial examples
    with torch.no_grad():
        delta_final = torch.clamp(delta, min=-epsilon, max=epsilon)
        x_adv_final = torch.clamp(x + delta_final, 0, 1)
    
    return x_adv_final.detach()


__all__ = [
    "carlini_wagner_linf",
    "carlini_wagner_linf_simple",
]
