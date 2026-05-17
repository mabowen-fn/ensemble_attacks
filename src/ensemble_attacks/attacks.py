import torch
import torch.nn.functional as F


def _validate_inputs(x, y, epsilon, alpha=None, iters=None):
    """Validate attack inputs."""
    assert x.dim() == 4, f"x must be 4D tensor (batch, channels, height, width), got {x.shape}"
    assert y.dim() == 1, f"y must be 1D tensor, got {y.shape}"
    assert x.shape[0] == y.shape[0], f"Batch size mismatch: {x.shape[0]} vs {y.shape[0]}"
    assert x.min() >= 0 and x.max() <= 1, f"x must be in [0, 1], got [{x.min():.4f}, {x.max():.4f}]"
    assert epsilon > 0, f"epsilon must be positive, got {epsilon}"
    
    if alpha is not None:
        assert alpha > 0, f"alpha must be positive, got {alpha}"
        assert alpha <= epsilon, f"alpha ({alpha}) should not exceed epsilon ({epsilon})"
    
    if iters is not None:
        assert iters > 0, f"iters must be positive, got {iters}"


def fgsm_attack(model, x, y, epsilon):
    """
    Fast Gradient Sign Method attack (single-step).
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32], must be in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        
    Returns:
        Adversarial examples [batch, 3, 32, 32], in [0, 1]
    """
    _validate_inputs(x, y, epsilon)
    
    x_adv = x.clone().detach().to(dtype=torch.float32).requires_grad_(True)

    logits = model(x_adv)
    loss = F.cross_entropy(logits, y)

    model.zero_grad()
    loss.backward()

    grad_sign = x_adv.grad.sign()
    x_adv = x_adv + epsilon * grad_sign
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()


def bim_attack(model, x, y, epsilon, alpha, iters):
    """
    Basic Iterative Method attack (iterative single-step).
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32], must be in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        Adversarial examples [batch, 3, 32, 32], in [0, 1]
    """
    _validate_inputs(x, y, epsilon, alpha, iters)
    
    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad_sign

        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv


def pgd_attack(model, x, y, epsilon, alpha, iters, random_start=True):
    """
    Projected Gradient Descent attack (strongest iterative attack).
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32], must be in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        alpha: Step size
        iters: Number of iterations
        random_start: Whether to start from random perturbation
        
    Returns:
        Adversarial examples [batch, 3, 32, 32], in [0, 1]
    """
    _validate_inputs(x, y, epsilon, alpha, iters)
    
    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad_(True)

        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)

        model.zero_grad()
        loss.backward()

        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad_sign

        delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + delta, 0, 1).detach()

    return x_adv


def mean_ensemble_attack(model, x, y, epsilon, alpha, iters):
    """
    Mean Ensemble Attack (MEA) = average of adversarial examples from FGSM, PGD, BIM.
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32], must be in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        alpha: Step size
        iters: Number of iterations
        
    Returns:
        Adversarial examples [batch, 3, 32, 32], in [0, 1]
    """
    _validate_inputs(x, y, epsilon, alpha, iters)
    
    x_fgsm = fgsm_attack(model, x, y, epsilon)
    x_pgd = pgd_attack(model, x, y, epsilon, alpha, iters)
    x_bim = bim_attack(model, x, y, epsilon, alpha, iters)

    x_ens = (x_fgsm + x_pgd + x_bim) / 3.0
    x_ens = torch.clamp(x_ens, 0, 1)
    return x_ens.detach()


def weighted_ensemble_attack(model, x, y, epsilon, alpha, iters,
                            w_fgsm=0.4, w_pgd=0.3, w_bim=0.3):
    """
    Weighted Ensemble Attack (WEA) = weighted combination of adversarial examples.
    
    Args:
        model: Target neural network
        x: Input images [batch, 3, 32, 32], must be in [0, 1]
        y: True labels [batch]
        epsilon: Perturbation budget
        alpha: Step size
        iters: Number of iterations
        w_fgsm, w_pgd, w_bim: Weights (must sum to 1.0)
        
    Returns:
        Adversarial examples [batch, 3, 32, 32], in [0, 1]
    """
    _validate_inputs(x, y, epsilon, alpha, iters)
    
    weight_sum = w_fgsm + w_pgd + w_bim
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights must sum to 1.0, got {weight_sum}"

    x_fgsm = fgsm_attack(model, x, y, epsilon)
    x_pgd = pgd_attack(model, x, y, epsilon, alpha, iters)
    x_bim = bim_attack(model, x, y, epsilon, alpha, iters)

    x_ens = w_fgsm * x_fgsm + w_pgd * x_pgd + w_bim * x_bim
    x_ens = torch.clamp(x_ens, 0, 1)
    return x_ens.detach()
