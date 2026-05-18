import os
import random
import numpy as np
import torch


def get_device(prefer_cuda: bool = True):
    """
    Return the best available device.
    Preference order: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The selected device.
        
    Raises:
        RuntimeError: If CUDA is requested but not available.
    """
    if prefer_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but not available. "
                f"CUDA available: {torch.cuda.is_available()}, "
                f"PyTorch version: {torch.__version__}"
            )
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """
    Return detailed device information for logging/debugging.
    """
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info["current_device"] = torch.cuda.current_device()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = torch.cuda.get_device_capability(0)
        info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Capture current memory usage
        info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / 1e9
        info["reserved_memory_gb"] = torch.cuda.memory_reserved(0) / 1e9
        
        try:
            torch.cuda.empty_cache()
            info["cache_cleared"] = True
        except Exception:
            info["cache_cleared"] = False
    
    info["mps_available"] = torch.backends.mps.is_available()
    info["cpu_count"] = os.cpu_count() or 1
    
    return info


def warmup_device(device: torch.device, iterations: int = 3):
    """
    Warmup the device with dummy forward passes to stabilize timings.
    Useful for GPU initialization on first use.
    """
    if device.type != "cuda":
        return
    
    try:
        dummy_model = torch.nn.Linear(128, 10).to(device)
        dummy_input = torch.randn(32, 128, device=device)
        
        for _ in range(iterations):
            with torch.no_grad():
                dummy_model(dummy_input)
        
        torch.cuda.synchronize()
        del dummy_model, dummy_input
        torch.cuda.empty_cache()
    except Exception as e:
        # Non-fatal, just log and continue
        pass


def set_random_seed(seed: int | None = None, deterministic: bool = False):
    """
    Configure random seeds for reproducibility. If `deterministic` is True,
    attempts to enable deterministic algorithms (may reduce performance).
    
    Args:
        seed: Random seed value. If None, skips seed setting.
        deterministic: If True, enables deterministic algorithms.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        # For GPU runs, prefer benchmark mode for performance unless
        # deterministic behaviour is requested.
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


__all__ = ["get_device", "get_device_info", "warmup_device", "set_random_seed"]
