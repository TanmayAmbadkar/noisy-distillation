import torch

def get_device(cfg_device: str = "auto") -> torch.device:
    """
    Returns the appropriate torch device based on the configuration or availability.
    
    Priority when cfg_device is "auto":
    1. CUDA
    2. MPS (for Apple Silicon)
    3. CPU fallback
    
    If a specific device is provided (e.g., "cuda:1"), it returns that device.
    """
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            try:
                # Some operations might fail on MPS, but we try to use it if available
                return torch.device("mps")
            except Exception:
                return torch.device("cpu")
        else:
            return torch.device("cpu")
    
    return torch.device(cfg_device)
