import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, device, log_fn=None):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        loader: DataLoader for training data
        optimizer: Optimizer instance
        device: torch.device to use
        log_fn: Optional logging function
        
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Batch-level metrics tracking
    batch_count = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Ensure correct dtype and device
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        assert x.ndim == 4 and x.shape[1] == 3, f"Invalid input shape: {x.shape}"
        assert x.min() >= 0 and x.max() <= 1, "Input must be in [0, 1] range"

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
        batch_count += 1
        
        # Log batch-level metrics if logging function provided
        if log_fn is not None and batch_idx % 10 == 0:
            batch_acc = (preds == y).sum().item() / y.size(0)
            log_fn(f"Batch {batch_idx}: loss={loss.item():.4f}, acc={batch_acc:.4f}")

    avg_loss = total_loss / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device, log_fn=None):
    """
    Evaluate model on validation/test data.
    
    Args:
        model: Neural network model
        loader: DataLoader for validation data
        device: torch.device to use
        log_fn: Optional logging function
        
    Returns:
        float: Accuracy
    """
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="Evaluating", leave=False)):
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        assert x.min() >= 0 and x.max() <= 1, "Input must be in [0, 1] range"

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += x.size(0)

    accuracy = correct / total if total > 0 else 0
    return accuracy
