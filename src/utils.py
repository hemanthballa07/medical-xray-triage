"""
Utility functions for the Medical X-ray Triage project.

This module contains helper functions for seeding, metrics calculation, and plotting.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import os
import json
from pathlib import Path


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics.
    
    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        threshold (float): Classification threshold
    
    Returns:
        dict: Dictionary containing computed metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        'auroc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    # Sensitivity and Specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def plot_roc_curve(y_true, y_prob, save_path=None, title="ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True binary labels
        y_prob (array-like): Predicted probabilities
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        y_true (array-like): True binary labels
        y_pred (array-like): Predicted binary labels
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None, title="Training History"):
    """
    Plot training history.
    
    Args:
        history (dict): Training history containing losses and metrics
        save_path (str): Path to save the plot
        title (str): Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUROC
    axes[0, 1].plot(history['train_auroc'], label='Train')
    axes[0, 1].plot(history['val_auroc'], label='Validation')
    axes[0, 1].set_title('AUROC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train')
    axes[1, 0].plot(history['val_f1'], label='Validation')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 1].plot(history['train_acc'], label='Train')
    axes[1, 1].plot(history['val_acc'], label='Validation')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to: {save_path}")
    
    plt.show()


def save_metrics(metrics, save_path):
    """
    Save metrics to JSON file.
    
    Args:
        metrics (dict): Metrics dictionary
        save_path (str): Path to save the metrics
    """
    # Cast to native types before json dump
    def safe_convert(obj):
        if isinstance(obj, dict):
            return {k: safe_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_convert(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    safe = safe_convert(metrics)
    
    with open(save_path, 'w') as f:
        json.dump(safe, f, indent=2)
    
    print(f"Metrics saved to: {save_path}")


def load_metrics(load_path):
    """
    Load metrics from JSON file.
    
    Args:
        load_path (str): Path to load the metrics from
    
    Returns:
        dict: Loaded metrics dictionary
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def print_metrics_table(metrics, title="Model Performance"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics (dict): Metrics dictionary
        title (str): Table title
    """
    print(f"\n{title}")
    print("=" * 50)
    
    # Define metric names and formatting
    metric_names = {
        'auroc': 'AUROC',
        'f1': 'F1 Score',
        'precision': 'Precision',
        'recall': 'Recall',
        'sensitivity': 'Sensitivity',
        'specificity': 'Specificity',
        'threshold': 'Threshold'
    }
    
    for key, display_name in metric_names.items():
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                print(f"{display_name:15}: {value:.4f}")
            else:
                print(f"{display_name:15}: {value}")
    
    print("=" * 50)


def get_device():
    """
    Get the best available device.
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    return device


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """
    Format time in seconds to human readable format.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test seeding
    seed_everything(42)
    print("✓ Seeding works")
    
    # Test device detection
    device = get_device()
    print(f"✓ Device detection: {device}")
    
    # Test metrics computation with dummy data
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    
    metrics = compute_metrics(y_true, y_prob)
    print("✓ Metrics computation works")
    print_metrics_table(metrics, "Test Metrics")
    
    print("All utility functions working correctly!")
