"""
Failure case analysis with visualizations.

This module provides functions to analyze and visualize model failure cases
(false positives and false negatives).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset


def identify_failure_cases(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    max_cases: int = 10
) -> Dict[str, np.ndarray]:
    """
    Identify false positive and false negative cases.
    
    Args:
        predictions: Predicted probabilities
        labels: True labels
        threshold: Classification threshold
        max_cases: Maximum number of cases to return
    
    Returns:
        dict: Dictionary with 'false_positives' and 'false_negatives' indices
    """
    y_pred = (predictions >= threshold).astype(int)
    
    # False positives: predicted positive but actually negative
    fp_indices = np.where((labels == 0) & (y_pred == 1))[0]
    
    # False negatives: predicted negative but actually positive
    fn_indices = np.where((labels == 1) & (y_pred == 0))[0]
    
    # Sort by confidence (most confident errors first)
    fp_sorted = fp_indices[np.argsort(predictions[fp_indices])[::-1]][:max_cases]
    fn_sorted = fn_indices[np.argsort(predictions[fn_indices])][:max_cases]
    
    return {
        'false_positives': fp_sorted,
        'false_negatives': fn_sorted
    }


def visualize_failure_cases(
    dataset: Dataset,
    failure_indices: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    failure_type: str,
    save_path: str,
    max_cases: int = 9
):
    """
    Visualize failure cases in a grid.
    
    Args:
        dataset: Dataset object
        failure_indices: Indices of failure cases
        predictions: Predicted probabilities
        labels: True labels
        failure_type: 'false_positives' or 'false_negatives'
        save_path: Path to save the visualization
        max_cases: Maximum number of cases to visualize
    """
    n_cases = min(len(failure_indices), max_cases)
    if n_cases == 0:
        print(f"No {failure_type} cases to visualize")
        return
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_cases + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, case_idx in enumerate(failure_indices[:n_cases]):
        ax = axes[idx]
        
        try:
            # Get image and label
            image, label = dataset[case_idx]
            
            # Convert tensor to numpy for display
            if isinstance(image, torch.Tensor):
                # Denormalize ImageNet normalization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_np = image * std + mean
                image_np = image_np.clamp(0, 1)
                image_np = image_np.permute(1, 2, 0).numpy()
            else:
                image_np = np.array(image)
                if image_np.max() > 1:
                    image_np = image_np / 255.0
            
            # Display image
            ax.imshow(image_np, cmap='gray' if len(image_np.shape) == 2 else None)
            
            # Title with prediction info
            true_label = "Pneumonia" if label == 1 else "Normal"
            pred_prob = predictions[case_idx]
            pred_label = "Pneumonia" if pred_prob >= 0.5 else "Normal"
            
            title = f"Case {case_idx}\n"
            title += f"True: {true_label}\n"
            title += f"Pred: {pred_label} ({pred_prob:.3f})"
            
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error loading\ncase {case_idx}:\n{str(e)}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Case {case_idx} (Error)")
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_cases, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{failure_type.replace("_", " ").title()} Cases', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Failure cases visualization saved to: {save_path}")
    
    plt.close()


def analyze_and_visualize_failures(
    model: torch.nn.Module,
    dataset: Dataset,
    predictions: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: str = "./results",
    max_cases: int = 9
) -> Dict:
    """
    Complete failure analysis with visualizations.
    
    Args:
        model: Trained model
        dataset: Test dataset
        predictions: Predicted probabilities
        labels: True labels
        device: Device to run on
        threshold: Classification threshold
        output_dir: Directory to save visualizations
        max_cases: Maximum cases to visualize
    
    Returns:
        dict: Analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify failure cases
    failures = identify_failure_cases(predictions, labels, threshold, max_cases)
    
    # Visualize false positives
    if len(failures['false_positives']) > 0:
        fp_path = os.path.join(output_dir, 'failure_cases_false_positives.png')
        visualize_failure_cases(
            dataset, failures['false_positives'], predictions, labels,
            'false_positives', fp_path, max_cases
        )
    
    # Visualize false negatives
    if len(failures['false_negatives']) > 0:
        fn_path = os.path.join(output_dir, 'failure_cases_false_negatives.png')
        visualize_failure_cases(
            dataset, failures['false_negatives'], predictions, labels,
            'false_negatives', fn_path, max_cases
        )
    
    # Summary statistics
    fp_probs = predictions[failures['false_positives']] if len(failures['false_positives']) > 0 else np.array([])
    fn_probs = predictions[failures['false_negatives']] if len(failures['false_negatives']) > 0 else np.array([])
    
    analysis = {
        'false_positives': {
            'count': len(failures['false_positives']),
            'mean_confidence': float(np.mean(fp_probs)) if len(fp_probs) > 0 else 0.0,
            'std_confidence': float(np.std(fp_probs)) if len(fp_probs) > 0 else 0.0,
        },
        'false_negatives': {
            'count': len(failures['false_negatives']),
            'mean_confidence': float(np.mean(fn_probs)) if len(fn_probs) > 0 else 0.0,
            'std_confidence': float(np.std(fn_probs)) if len(fn_probs) > 0 else 0.0,
        }
    }
    
    return analysis

