"""
Evaluation script for the Medical X-ray Triage project.

This script evaluates trained models on test data and generates comprehensive
metrics including ROC curves, confusion matrices, and classification reports.
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from data import create_data_loaders, get_simple_data_loader, create_pre_split_data_loaders, ChestXrayDataset, get_transforms
from model import create_model
from utils import (
    seed_everything, compute_metrics, print_metrics_table,
    plot_roc_curve, plot_confusion_matrix, save_metrics,
    get_device, format_time
)


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on
        threshold: Classification threshold
    
    Returns:
        dict: Evaluation metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_logits = []
    
    print("Evaluating model on test data...")
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluation")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            
            # Store results
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.squeeze().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Batch': f'{batch_idx+1}/{len(test_loader)}'
            })
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Compute metrics
    metrics = compute_metrics(all_labels, all_predictions, threshold=threshold)
    
    # Add additional metrics
    results = {
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'metrics': metrics,
        'threshold': threshold
    }
    
    return results


def generate_evaluation_plots(results, save_dir):
    """
    Generate and save evaluation plots.
    
    Args:
        results: Evaluation results dictionary
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ROC Curve
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plot_roc_curve(
        results['labels'], 
        results['predictions'],
        save_path=roc_path,
        title="ROC Curve - Test Set"
    )
    
    # Confusion Matrix
    y_pred = (results['predictions'] >= results['threshold']).astype(int)
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        results['labels'],
        y_pred,
        save_path=cm_path,
        title="Confusion Matrix - Test Set"
    )


def print_detailed_report(results):
    """
    Print detailed evaluation report.
    
    Args:
        results: Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("DETAILED EVALUATION REPORT")
    print("="*60)
    
    # Basic metrics
    metrics = results['metrics']
    print_metrics_table(metrics, "Test Set Performance")
    
    # Additional statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(results['labels'])}")
    print(f"  Normal samples: {np.sum(results['labels'] == 0)}")
    print(f"  Abnormal samples: {np.sum(results['labels'] == 1)}")
    print(f"  Classification threshold: {results['threshold']}")
    
    # Prediction statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean predicted probability: {np.mean(results['predictions']):.4f}")
    print(f"  Std predicted probability: {np.std(results['predictions']):.4f}")
    print(f"  Min predicted probability: {np.min(results['predictions']):.4f}")
    print(f"  Max predicted probability: {np.max(results['predictions']):.4f}")
    
    # Class-wise performance
    y_pred = (results['predictions'] >= results['threshold']).astype(int)
    
    print(f"\nClass-wise Performance:")
    print(f"  True Negatives: {np.sum((results['labels'] == 0) & (y_pred == 0))}")
    print(f"  False Positives: {np.sum((results['labels'] == 0) & (y_pred == 1))}")
    print(f"  False Negatives: {np.sum((results['labels'] == 1) & (y_pred == 0))}")
    print(f"  True Positives: {np.sum((results['labels'] == 1) & (y_pred == 1))}")
    
    # Confidence analysis
    correct_predictions = (y_pred == results['labels'])
    correct_probs = results['predictions'][correct_predictions]
    incorrect_probs = results['predictions'][~correct_predictions]
    
    if len(correct_probs) > 0 and len(incorrect_probs) > 0:
        print(f"\nConfidence Analysis:")
        print(f"  Mean confidence (correct): {np.mean(correct_probs):.4f}")
        print(f"  Mean confidence (incorrect): {np.mean(incorrect_probs):.4f}")
    
    print("="*60)


def find_optimal_threshold(results):
    """
    Find optimal classification threshold using Youden's J statistic.
    
    Args:
        results: Evaluation results dictionary
    
    Returns:
        float: Optimal threshold
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, thresholds = roc_curve(results['labels'], results['predictions'])
    
    # Calculate Youden's J statistic (TPR - FPR) for each threshold
    j_scores = tpr - fpr
    
    # Find threshold with maximum J score
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal Threshold Analysis:")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Youden's J score: {j_scores[optimal_idx]:.4f}")
    print(f"  TPR at optimal: {tpr[optimal_idx]:.4f}")
    print(f"  FPR at optimal: {fpr[optimal_idx]:.4f}")
    
    return optimal_threshold


def main():
    """Main evaluation function."""
    # Get configuration
    config = get_config()
    
    print("Medical X-ray Triage Evaluation")
    print("=" * 50)
    print(f"Model path: {config['model_path']}")
    print(f"Data: {config['data_dir']}")
    print(f"Device: {config['device']}")
    print()
    
    # Set seed for reproducibility
    seed_everything(config['seed'])
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found at {config['model_path']}")
        print("Please train a model first using: python src/train.py")
        return
    
    # Check if sample data exists
    if not os.path.exists(config['labels_path']):
        print("Sample data not found. Generating sample data...")
        from make_sample_data import create_sample_dataset
        create_sample_dataset(config['data_dir'])
        print()
    
    # Get device
    if config.get('device', 'auto') == 'auto':
        device = get_device()
    else:
        device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model = create_model(config['model_name'], num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(config['model_path'], map_location="cpu"))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Model: {config['model_name']}")
    print(f"  Epoch: Unknown (state_dict only)")
    
    # Create data loaders
    print("Loading test data...")
    
    # For small datasets, use simple data loader
    # Check if this is a pre-split NIH dataset
    train_labels_path = os.path.join(config['data_dir'], 'train_labels.csv')
    is_nih_dataset = os.path.exists(train_labels_path)
    
    if is_nih_dataset:
        print("NIH dataset detected (pre-split structure), using re-split test set for evaluation")
        # Use create_pre_split_data_loaders to get properly re-split test loader
        # This ensures we use the same 70/15/15 split as training
        _, _, test_loader, _ = create_pre_split_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers'],
            use_weighted_sampling=False  # Not needed for test set
        )
        print(f"Test dataset size: {len(test_loader.dataset)}")
    elif os.path.exists(config['labels_path']):
        import pandas as pd
        labels_df = pd.read_csv(config['labels_path'])
        if len(labels_df) <= 10:  # Small dataset, use simple loader
            print("Small dataset detected, using simple data loader")
            test_loader, dataset = get_simple_data_loader(
                labels_path=config['labels_path'],
                images_dir=config['images_dir'],
                batch_size=config['batch_size'],
                img_size=config['img_size'],
                is_training=False,
                num_workers=config['num_workers']
            )
        else:
            train_loader, val_loader, test_loader, class_weights = create_data_loaders(
                labels_path=config['labels_path'],
                images_dir=config['images_dir'],
                batch_size=config['batch_size'],
                img_size=config['img_size'],
                num_workers=config['num_workers']
            )
    else:
        raise FileNotFoundError(f"Labels file not found: {config['labels_path']}")
    
    print(f"Test batches: {len(test_loader)}")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, config['threshold'])
    
    # Print detailed report
    print_detailed_report(results)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(results)
    
    # Evaluate with optimal threshold
    optimal_results = evaluate_model(model, test_loader, device, optimal_threshold)
    print(f"\nPerformance with optimal threshold ({optimal_threshold:.4f}):")
    print_metrics_table(optimal_results['metrics'], "Optimal Threshold Performance")
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    generate_evaluation_plots(results, config['output_dir'])
    
    # Save results
    evaluation_results = {
        'default_threshold_results': results,
        'optimal_threshold_results': optimal_results,
        'optimal_threshold': optimal_threshold,
        'config': config,
        'model_name': config['model_name']
    }
    
    results_path = os.path.join(config['output_dir'], 'evaluation_results.json')
    save_metrics(evaluation_results, results_path)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {results_path}")
    print(f"Plots saved to: {config['output_dir']}")
    
    # Print one-line summary
    print(f"SUMMARY: AUROC: {results['metrics']['auroc']:.4f}, F1: {results['metrics']['f1']:.4f}, Optimal Threshold: {optimal_threshold:.4f}")


if __name__ == "__main__":
    main()
