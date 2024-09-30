import torch
import torch.nn.functional as F

def safe_logits_to_probs(logits):
    """
    Safely convert logits to probabilities.
    
    Args:
        logits (torch.Tensor): The input logits tensor.
    
    Returns:
        torch.Tensor: A tensor of probabilities.
    """
    # Subtract max for numerical stability
    logits = logits - torch.max(logits)
    probs = F.softmax(logits, dim=-1)
    # Add small epsilon to avoid zero probabilities
    probs = probs + 1e-10
    return probs / probs.sum()