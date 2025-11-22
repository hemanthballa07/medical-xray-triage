"""
Bootstrap confidence intervals for evaluation metrics.

This module provides functions to compute bootstrapped confidence intervals
for metrics like AUROC, F1, sensitivity, and specificity.
"""

import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from tqdm import tqdm


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    threshold: float = 0.5,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrapped confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric_func: Function to compute metric (e.g., roc_auc_score, f1_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        threshold: Classification threshold for binary metrics
        random_seed: Random seed for reproducibility
    
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in tqdm(range(n_bootstrap), desc="Bootstrapping", leave=False):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        try:
            if metric_func == roc_auc_score:
                score = metric_func(y_true_boot, y_prob_boot)
            else:
                y_pred_boot = (y_prob_boot >= threshold).astype(int)
                score = metric_func(y_true_boot, y_pred_boot, zero_division=0)
            bootstrap_scores.append(score)
        except ValueError:
            # Skip if metric can't be computed (e.g., only one class in bootstrap)
            continue
    
    if len(bootstrap_scores) == 0:
        return 0.0, 0.0, 0.0
    
    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
    upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
    
    return float(mean_score), float(lower), float(upper)


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrapped confidence intervals for all metrics.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        threshold: Classification threshold
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level
        random_seed: Random seed
    
    Returns:
        dict: Dictionary with bootstrapped metrics and confidence intervals
    """
    print(f"Computing bootstrapped confidence intervals (n={n_bootstrap})...")
    
    results = {}
    
    # AUROC
    print("  Bootstrapping AUROC...")
    auroc_mean, auroc_lower, auroc_upper = bootstrap_metric(
        y_true, y_prob, roc_auc_score, n_bootstrap, confidence_level, threshold, random_seed
    )
    results['auroc'] = {
        'mean': auroc_mean,
        'lower': auroc_lower,
        'upper': auroc_upper,
        'ci': f"[{auroc_lower:.4f}, {auroc_upper:.4f}]"
    }
    
    # F1 Score
    print("  Bootstrapping F1 Score...")
    f1_mean, f1_lower, f1_upper = bootstrap_metric(
        y_true, y_prob, f1_score, n_bootstrap, confidence_level, threshold, random_seed
    )
    results['f1'] = {
        'mean': f1_mean,
        'lower': f1_lower,
        'upper': f1_upper,
        'ci': f"[{f1_lower:.4f}, {f1_upper:.4f}]"
    }
    
    # Precision
    print("  Bootstrapping Precision...")
    precision_mean, precision_lower, precision_upper = bootstrap_metric(
        y_true, y_prob, precision_score, n_bootstrap, confidence_level, threshold, random_seed
    )
    results['precision'] = {
        'mean': precision_mean,
        'lower': precision_lower,
        'upper': precision_upper,
        'ci': f"[{precision_lower:.4f}, {precision_upper:.4f}]"
    }
    
    # Recall (Sensitivity)
    print("  Bootstrapping Recall (Sensitivity)...")
    recall_mean, recall_lower, recall_upper = bootstrap_metric(
        y_true, y_prob, recall_score, n_bootstrap, confidence_level, threshold, random_seed
    )
    results['recall'] = {
        'mean': recall_mean,
        'lower': recall_lower,
        'upper': recall_upper,
        'ci': f"[{recall_lower:.4f}, {recall_upper:.4f}]"
    }
    results['sensitivity'] = results['recall']  # Alias
    
    return results


def plot_bootstrap_distributions(
    bootstrap_results: Dict[str, Dict[str, float]],
    save_path: str = None
):
    """
    Plot bootstrap distributions for metrics.
    
    Args:
        bootstrap_results: Results from bootstrap_all_metrics
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    n_metrics = len(bootstrap_results)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, (metric_name, metric_data) in enumerate(bootstrap_results.items()):
        ax = axes[idx]
        
        # Create a simple bar plot with error bars
        mean = metric_data['mean']
        lower = metric_data['lower']
        upper = metric_data['upper']
        
        ax.bar([0], [mean], yerr=[[mean - lower], [upper - mean]], 
               capsize=10, alpha=0.7, color='steelblue')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title(f'{metric_name.upper()}\n95% CI: {metric_data["ci"]}')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Bootstrap distributions plot saved to: {save_path}")
    
    return fig

