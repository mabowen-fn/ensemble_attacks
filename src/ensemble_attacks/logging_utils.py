import csv
import os
import json
import sys
from datetime import datetime
import platform
from typing import Any, Dict


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def now_timestamp():
    """Return current timestamp in ISO format."""
    return datetime.now().isoformat(timespec="seconds")


def append_csv_row(csv_path: str, row: dict):
    """Append a row to CSV file, creating it with headers if needed."""
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)


class ExperimentLogger:
    """Structured experiment logging for reproducibility and debugging."""
    
    def __init__(self, log_dir: str = "outputs/logs"):
        self.log_dir = log_dir
        ensure_dir(log_dir)
        
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join(log_dir, self.exp_id)
        ensure_dir(self.exp_dir)
        
        self.metrics_csv = os.path.join(self.exp_dir, "metrics.csv")
        self.metadata_file = os.path.join(self.exp_dir, "metadata.json")
        self.log_file = os.path.join(self.exp_dir, "experiment.log")
        
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize experiment metadata file."""
        metadata = {
            "exp_id": self.exp_id,
            "timestamp": now_timestamp(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
        }
        
        # Try to capture torch/device info
        try:
            import torch
            metadata["pytorch_version"] = torch.__version__
            metadata["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                metadata["cuda_version"] = torch.version.cuda
                metadata["device_name"] = torch.cuda.get_device_name(0)
                metadata["device_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
        
        self.metadata = metadata
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def log_metric(self, row: Dict[str, Any]):
        """Log a metric row (appended to CSV)."""
        row["timestamp"] = now_timestamp()
        append_csv_row(self.metrics_csv, row)
    
    def log_text(self, message: str, level: str = "INFO"):
        """Log a text message to file and stdout."""
        timestamp = now_timestamp()
        formatted = f"[{timestamp}] [{level}] {message}"
        
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")
    
    def get_exp_dir(self) -> str:
        """Return the experiment directory path."""
        return self.exp_dir


def print_device_info(device_info: dict):
    """Pretty-print device information."""
    print("\n" + "="*60)
    print("DEVICE INFORMATION")
    print("="*60)
    
    print(f"PyTorch Version: {device_info.get('pytorch_version', 'N/A')}")
    print(f"CUDA Available: {device_info.get('cuda_available', False)}")
    
    if device_info.get('cuda_available'):
        print(f"CUDA Version: {device_info.get('cuda_version', 'N/A')}")
        print(f"cuDNN Version: {device_info.get('cudnn_version', 'N/A')}")
        print(f"GPU Device Count: {device_info.get('device_count', 0)}")
        print(f"Current Device: {device_info.get('current_device', 0)}")
        print(f"Device Name: {device_info.get('device_name', 'N/A')}")
        print(f"Device Capability: {device_info.get('device_capability', 'N/A')}")
        print(f"Total Memory: {device_info.get('total_memory_gb', 0):.2f} GB")
        print(f"Allocated Memory: {device_info.get('allocated_memory_gb', 0):.4f} GB")
        print(f"Reserved Memory: {device_info.get('reserved_memory_gb', 0):.4f} GB")
    
    print(f"MPS Available: {device_info.get('mps_available', False)}")
    print(f"CPU Count: {device_info.get('cpu_count', 0)}")
    print("="*60 + "\n")


__all__ = [
    "ensure_dir", 
    "now_timestamp", 
    "append_csv_row",
    "ExperimentLogger",
    "print_device_info",
]
