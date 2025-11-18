"""
Additional plotting utilities for evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, precision_recall_curve, 
    roc_auc_score, average_precision_score
)


def plot_precision_recall_curve(y_true, y_prob, save_path=None, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_score = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {ap_score:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
    
    return plt.gcf()


def plot_roc_vs_threshold(y_true, y_prob, save_path=None, title="ROC Metrics vs Threshold"):
    """
    Plot ROC metrics (TPR, FPR) and F1 score vs threshold.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        save_path: Path to save the plot
        title: Plot title
    """
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tprs.append(tpr)
        fprs.append(fpr)
        f1_scores.append(f1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate', color='black')
    ax1.plot(thresholds, tprs, 'b-', label='True Positive Rate (Sensitivity)', linewidth=2)
    ax1.plot(thresholds, fprs, 'r-', label='False Positive Rate', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='green')
    ax2.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper right')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC vs Threshold plot saved to: {save_path}")
    
    return fig


def plot_gradcam_comparison(original_image, gradcam_results, save_path=None):
    """
    Plot comparison of different Grad-CAM methods side-by-side.
    
    Args:
        original_image: Original input image
        gradcam_results: Dict mapping method names to (overlay_image, heatmap)
        save_path: Path to save the plot
    """
    n_methods = len(gradcam_results)
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4 * (n_methods + 1), 8))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Each Grad-CAM method
    for idx, (method_name, (overlay, heatmap)) in enumerate(gradcam_results.items(), 1):
        # Overlay
        axes[0, idx].imshow(overlay)
        axes[0, idx].set_title(f"{method_name}\nOverlay")
        axes[0, idx].axis('off')
        
        # Heatmap
        im = axes[1, idx].imshow(heatmap, cmap='jet')
        axes[1, idx].set_title(f"{method_name}\nHeatmap")
        axes[1, idx].axis('off')
        plt.colorbar(im, ax=axes[1, idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM comparison saved to: {save_path}")
    
    return fig

