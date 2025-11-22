"""
Audit module for subgroup metrics and fairness analysis.

This module computes performance metrics across different subgroups
to support Responsible AI goals and fairness analysis.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

from utils import compute_metrics


def compute_subgroup_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    subgroups: np.ndarray,
    subgroup_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict:
    """
    Compute metrics for each subgroup.
    
    Args:
        predictions: Predicted probabilities
        labels: True labels
        subgroups: Subgroup assignments (e.g., age groups, device types)
        subgroup_names: Names for each subgroup
        threshold: Classification threshold
    
    Returns:
        dict: Metrics for each subgroup
    """
    unique_subgroups = np.unique(subgroups)
    
    if subgroup_names is None:
        subgroup_names = [f"Subgroup_{i}" for i in range(len(unique_subgroups))]
    
    results = {}
    
    for subgroup_id, subgroup_name in zip(unique_subgroups, subgroup_names):
        mask = subgroups == subgroup_id
        subgroup_predictions = predictions[mask]
        subgroup_labels = labels[mask]
        
        if len(subgroup_labels) == 0:
            continue
        
        metrics = compute_metrics(subgroup_labels, subgroup_predictions, threshold)
        
        results[subgroup_name] = {
            'n_samples': int(np.sum(mask)),
            'metrics': metrics,
            'subgroup_id': int(subgroup_id)
        }
    
    return results


def analyze_fairness(
    subgroup_metrics: Dict,
    reference_subgroup: str = None
) -> Dict:
    """
    Analyze fairness across subgroups.
    
    Args:
        subgroup_metrics: Metrics for each subgroup
        reference_subgroup: Reference subgroup for comparison
    
    Returns:
        dict: Fairness analysis results
    """
    if reference_subgroup is None:
        # Use largest subgroup as reference
        reference_subgroup = max(subgroup_metrics.keys(), 
                               key=lambda k: subgroup_metrics[k]['n_samples'])
    
    if reference_subgroup not in subgroup_metrics:
        return {}
    
    ref_metrics = subgroup_metrics[reference_subgroup]['metrics']
    fairness_analysis = {
        'reference_subgroup': reference_subgroup,
        'comparisons': {}
    }
    
    for subgroup_name, subgroup_data in subgroup_metrics.items():
        if subgroup_name == reference_subgroup:
            continue
        
        metrics = subgroup_data['metrics']
        comparison = {
            'auroc_diff': metrics['auroc'] - ref_metrics['auroc'],
            'f1_diff': metrics['f1'] - ref_metrics['f1'],
            'sensitivity_diff': metrics['sensitivity'] - ref_metrics['sensitivity'],
            'specificity_diff': metrics['specificity'] - ref_metrics['specificity'],
            'relative_auroc': metrics['auroc'] / ref_metrics['auroc'] if ref_metrics['auroc'] > 0 else 0,
            'relative_f1': metrics['f1'] / ref_metrics['f1'] if ref_metrics['f1'] > 0 else 0,
        }
        
        fairness_analysis['comparisons'][subgroup_name] = comparison
    
    return fairness_analysis


def visualize_subgroup_metrics(
    subgroup_metrics: Dict,
    save_path: str = None
):
    """
    Visualize metrics across subgroups.
    
    Args:
        subgroup_metrics: Metrics for each subgroup
        save_path: Path to save the plot
    """
    subgroups = list(subgroup_metrics.keys())
    metrics_to_plot = ['auroc', 'f1', 'sensitivity', 'specificity']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [subgroup_metrics[sg]['metrics'][metric_name] for sg in subgroups]
        n_samples = [subgroup_metrics[sg]['n_samples'] for sg in subgroups]
        
        # Bar plot with sample size annotation
        bars = ax.bar(subgroups, values, alpha=0.7, color='steelblue')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f'{metric_name.upper()} by Subgroup')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size labels
        for bar, n in zip(bars, n_samples):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={n}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Subgroup metrics visualization saved to: {save_path}")
    
    return fig


def audit_model_performance(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    subgroups: np.ndarray,
    subgroup_names: Optional[List[str]] = None,
    threshold: float = 0.5,
    output_dir: str = "./results"
) -> Dict:
    """
    Complete audit of model performance across subgroups.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to run on
        subgroups: Subgroup assignments
        subgroup_names: Names for subgroups
        threshold: Classification threshold
        output_dir: Output directory
    
    Returns:
        dict: Complete audit results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute subgroup metrics
    subgroup_metrics = compute_subgroup_metrics(
        all_predictions, all_labels, subgroups, subgroup_names, threshold
    )
    
    # Analyze fairness
    fairness_analysis = analyze_fairness(subgroup_metrics)
    
    # Visualize
    viz_path = os.path.join(output_dir, 'subgroup_metrics.png')
    visualize_subgroup_metrics(subgroup_metrics, save_path=viz_path)
    
    # Create summary report
    audit_results = {
        'subgroup_metrics': {
            k: {
                'n_samples': v['n_samples'],
                'metrics': {mk: float(mv) for mk, mv in v['metrics'].items()}
            }
            for k, v in subgroup_metrics.items()
        },
        'fairness_analysis': fairness_analysis
    }
    
    # Print summary
    print("\n" + "="*60)
    print("AUDIT RESULTS - Subgroup Performance")
    print("="*60)
    
    for subgroup_name, data in subgroup_metrics.items():
        metrics = data['metrics']
        print(f"\n{subgroup_name} (n={data['n_samples']}):")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    if fairness_analysis:
        print(f"\nFairness Analysis (Reference: {fairness_analysis['reference_subgroup']}):")
        for subgroup, comparison in fairness_analysis['comparisons'].items():
            print(f"\n  {subgroup} vs Reference:")
            print(f"    AUROC difference: {comparison['auroc_diff']:+.4f}")
            print(f"    F1 difference: {comparison['f1_diff']:+.4f}")
            print(f"    Relative AUROC: {comparison['relative_auroc']:.2%}")
    
    return audit_results


# Example: Create synthetic subgroups based on prediction confidence
def create_confidence_subgroups(
    predictions: np.ndarray,
    n_groups: int = 3
) -> np.ndarray:
    """
    Create subgroups based on prediction confidence.
    
    Args:
        predictions: Predicted probabilities
        n_groups: Number of groups
    
    Returns:
        np.ndarray: Subgroup assignments
    """
    # Group by confidence level
    percentiles = np.linspace(0, 100, n_groups + 1)
    thresholds = np.percentile(predictions, percentiles)
    
    subgroups = np.zeros_like(predictions, dtype=int)
    for i in range(n_groups):
        mask = (predictions >= thresholds[i]) & (predictions < thresholds[i+1])
        if i == n_groups - 1:  # Include upper bound for last group
            mask = predictions >= thresholds[i]
        subgroups[mask] = i
    
    return subgroups

