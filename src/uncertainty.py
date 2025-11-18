"""
Uncertainty estimation for model predictions.

This module provides Monte-Carlo dropout and temperature scaling
for uncertainty quantification.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import scipy.stats as stats


class MonteCarloDropout(nn.Module):
    """
    Wrapper to enable dropout during inference for uncertainty estimation.
    """
    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        """
        Args:
            model: Trained PyTorch model
            dropout_rate: Dropout rate to use (should match training)
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers in the model."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active
                if hasattr(module, 'p'):
                    module.p = self.dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout enabled."""
        return self.model(x)


def monte_carlo_dropout_predict(
    model: nn.Module,
    input_tensor: torch.Tensor,
    n_samples: int = 100,
    dropout_rate: float = 0.1
) -> Tuple[float, float, np.ndarray]:
    """
    Perform Monte-Carlo dropout inference to estimate uncertainty.
    
    Args:
        model: Trained PyTorch model
        input_tensor: Input tensor (batch_size, channels, height, width)
        n_samples: Number of Monte-Carlo samples
        dropout_rate: Dropout rate to use
    
    Returns:
        tuple: (mean_prob, std_prob, all_probs)
    """
    mc_model = MonteCarloDropout(model, dropout_rate)
    mc_model.eval()
    
    all_probs = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            logits = mc_model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs.flatten())
    
    all_probs = np.array(all_probs)
    mean_prob = float(np.mean(all_probs))
    std_prob = float(np.std(all_probs))
    
    return mean_prob, std_prob, all_probs


def compute_confidence_interval(
    probs: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for predictions.
    
    Args:
        probs: Array of probability samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower = float(np.percentile(probs, 100 * alpha / 2))
    upper = float(np.percentile(probs, 100 * (1 - alpha / 2)))
    return lower, upper


def temperature_scaling_uncertainty(
    logits: torch.Tensor,
    temperature: float = 1.0
) -> Tuple[float, float]:
    """
    Apply temperature scaling and estimate uncertainty.
    
    Args:
        logits: Model logits
        temperature: Temperature parameter (higher = more uncertain)
    
    Returns:
        tuple: (calibrated_prob, uncertainty_estimate)
    """
    scaled_logits = logits / temperature
    prob = torch.sigmoid(scaled_logits).item()
    
    # Uncertainty estimate based on temperature
    # Higher temperature = higher uncertainty
    uncertainty = min(1.0, temperature / 2.0) if temperature > 1.0 else 0.0
    
    return prob, uncertainty

