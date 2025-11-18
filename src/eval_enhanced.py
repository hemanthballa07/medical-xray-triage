"""
Enhanced evaluation script for the Medical X-ray Triage project.

This script provides comprehensive evaluation including:
- Multiple threshold evaluation (default, optimal F1, operating threshold)
- Probability calibration (Temperature Scaling)
- Error analysis
- Robustness checks
- Metadata generation
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import platform
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, precision_recall_curve, f1_score,
    confusion_matrix, roc_auc_score
)
from sklearn.calibration import calibration_curve
import scipy.optimize

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from data import create_pre_split_data_loaders, ChestXrayDataset, get_transforms
from model import create_model
from utils import (
    seed_everything, compute_metrics, print_metrics_table,
    plot_roc_curve, plot_confusion_matrix, save_metrics,
    get_device, format_time
)


class TemperatureScaling:
    """Temperature Scaling for probability calibration."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Fit temperature parameter."""
        def eval_temperature(t):
            scaled_logits = logits / t
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, labels.float()
            )
            return loss.item()
        
        # Optimize temperature
        result = scipy.optimize.minimize_scalar(
            eval_temperature, bounds=(0.01, 10.0), method='bounded'
        )
        self.temperature = result.x
        return self
    
    def transform(self, logits):
        """Apply temperature scaling."""
        return logits / self.temperature


def evaluate_at_threshold(predictions, labels, threshold):
    """Evaluate metrics at a specific threshold."""
    y_pred = (predictions >= threshold).astype(int)
    metrics = compute_metrics(labels, predictions, threshold=threshold)
    
    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['confusion_matrix'] = {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    
    return metrics


def find_optimal_f1_threshold(predictions, labels):
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5


def find_operating_threshold(predictions, labels, min_specificity=0.93):
    """Find operating threshold that prioritizes recall while maintaining specificity."""
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    specificity = 1 - fpr
    
    # Find thresholds that meet minimum specificity
    valid_indices = np.where(specificity >= min_specificity)[0]
    
    if len(valid_indices) > 0:
        # Among valid thresholds, choose one with highest recall
        best_idx = valid_indices[np.argmax(tpr[valid_indices])]
        return thresholds[best_idx]
    else:
        # If no threshold meets requirement, use optimal F1
        return find_optimal_f1_threshold(predictions, labels)


def compute_ece(predictions, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def plot_reliability_diagram(predictions, labels, save_path=None, n_bins=10):
    """Plot reliability diagram for calibration assessment."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        labels, predictions, n_bins=n_bins, strategy='uniform'
    )
    
    ece = compute_ece(predictions, labels, n_bins=n_bins)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Reliability Diagram (ECE = {ece:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reliability diagram saved to: {save_path}")
    
    plt.close()


def analyze_errors(predictions, labels, threshold, image_paths=None):
    """Analyze false negatives and false positives."""
    y_pred = (predictions >= threshold).astype(int)
    
    fn_indices = np.where((labels == 1) & (y_pred == 0))[0]
    fp_indices = np.where((labels == 0) & (y_pred == 1))[0]
    
    analysis = {
        'false_negatives': {
            'count': int(len(fn_indices)),
            'mean_confidence': float(predictions[fn_indices].mean()) if len(fn_indices) > 0 else 0.0,
            'std_confidence': float(predictions[fn_indices].std()) if len(fn_indices) > 0 else 0.0,
            'min_confidence': float(predictions[fn_indices].min()) if len(fn_indices) > 0 else 0.0,
            'max_confidence': float(predictions[fn_indices].max()) if len(fn_indices) > 0 else 0.0,
        },
        'false_positives': {
            'count': int(len(fp_indices)),
            'mean_confidence': float(predictions[fp_indices].mean()) if len(fp_indices) > 0 else 0.0,
            'std_confidence': float(predictions[fp_indices].std()) if len(fp_indices) > 0 else 0.0,
            'min_confidence': float(predictions[fp_indices].min()) if len(fp_indices) > 0 else 0.0,
            'max_confidence': float(predictions[fp_indices].max()) if len(fp_indices) > 0 else 0.0,
        }
    }
    
    # Generate summary bullets
    summary = []
    if len(fn_indices) > 0:
        summary.append(f"False Negatives: {len(fn_indices)} cases with mean confidence {analysis['false_negatives']['mean_confidence']:.3f}")
    if len(fp_indices) > 0:
        summary.append(f"False Positives: {len(fp_indices)} cases with mean confidence {analysis['false_positives']['mean_confidence']:.3f}")
    
    if len(fn_indices) > len(fp_indices) * 1.5:
        summary.append("Class imbalance observed: More false negatives than false positives")
    elif len(fp_indices) > len(fn_indices) * 1.5:
        summary.append("Class imbalance observed: More false positives than false negatives")
    
    if len(fn_indices) > 0 and predictions[fn_indices].mean() < 0.3:
        summary.append("False negatives tend to have low confidence scores, suggesting difficult cases")
    
    if len(fp_indices) > 0 and predictions[fp_indices].mean() > 0.7:
        summary.append("False positives tend to have high confidence, suggesting model overconfidence")
    
    analysis['summary'] = summary
    return analysis


def evaluate_with_augmentations(model, test_dataset, device, 
                                img_size, batch_size, num_workers):
    """Evaluate model with stronger augmentations for robustness testing."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    
    # Create stronger augmentations
    augmentations = {
        'stronger_crop': transforms.Compose([
            transforms.Resize((img_size + 64, img_size + 64)),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'stronger_rotation': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'stronger_color_jitter': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    results = {}
    
    # Get the underlying dataset from the Subset and save original transform
    if hasattr(test_dataset, 'dataset'):
        base_dataset = test_dataset.dataset
        original_transform = base_dataset.transform
    else:
        base_dataset = test_dataset
        original_transform = base_dataset.transform if hasattr(base_dataset, 'transform') else None
    
    for aug_name, aug_transform in augmentations.items():
        # Temporarily set the transform
        base_dataset.transform = aug_transform
        
        # Create loader (reuse the same subset/dataset)
        aug_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        # Evaluate
        all_predictions = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for images, labels in aug_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits.squeeze())
                all_predictions.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = compute_metrics(all_labels, all_predictions, threshold=0.5)
        results[aug_name] = {
            'auroc': float(metrics['auroc']),
            'f1': float(metrics['f1']),
            'recall': float(metrics['recall'])
        }
    
    # Restore original transform
    if original_transform is not None:
        base_dataset.transform = original_transform
    
    return results


def create_metadata(config, model):
    """Create metadata JSON for reproducibility."""
    import torch
    
    metadata = {
        'random_seed': config.get('seed', 1337),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'mps_available': hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
        'model_architecture': config.get('model_name', 'resnet18'),
        'training_hyperparameters': {
            'batch_size': config.get('batch_size', 8),
            'learning_rate': config.get('lr', 0.0001),
            'weight_decay': config.get('weight_decay', 0.0001),
            'epochs': config.get('epochs', 25),
            'patience': config.get('patience', 8),
            'img_size': config.get('img_size', 320),
            'pretrained': config.get('pretrained', True)
        },
        'platform': {
            'system': platform.system(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    }
    
    return metadata


def main():
    """Main enhanced evaluation function."""
    config = get_config()
    
    print("Enhanced Medical X-ray Triage Evaluation")
    print("=" * 60)
    print(f"Model path: {config['model_path']}")
    print(f"Data: {config['data_dir']}")
    print(f"Device: {config['device']}")
    print()
    
    seed_everything(config['seed'])
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found at {config['model_path']}")
        return
    
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
    print(f"Model loaded: {config['model_name']}")
    
    # Load test data using the same re-splitting logic as training
    print("Loading test data...")
    train_labels_path = os.path.join(config['data_dir'], 'train_labels.csv')
    is_nih_dataset = os.path.exists(train_labels_path)
    
    if not is_nih_dataset:
        print("Error: Enhanced evaluation requires pre-split dataset structure")
        return
    
    # Use create_pre_split_data_loaders to get properly re-split test loader
    # This ensures we use the same 70/15/15 split as training
    _, _, test_loader, _ = create_pre_split_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        use_weighted_sampling=False  # Not needed for test set
    )
    
    print(f"Test batches: {len(test_loader)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Get predictions
    print("\nGetting model predictions...")
    all_predictions = []
    all_labels = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.squeeze().cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Save predictions and labels for later plot generation
    predictions_path = os.path.join(config['output_dir'], 'predictions.npz')
    np.savez(predictions_path, 
             predictions=all_predictions, 
             labels=all_labels, 
             logits=all_logits)
    print(f"Predictions saved to: {predictions_path}")
    
    # 1. THRESHOLD SELECTION
    print("\n" + "="*60)
    print("1. THRESHOLD SELECTION")
    print("="*60)
    
    # Default threshold
    default_threshold = 0.5
    default_metrics = evaluate_at_threshold(all_predictions, all_labels, default_threshold)
    print(f"\nDefault Threshold (0.50):")
    print_metrics_table(default_metrics, "Metrics")
    
    # Optimal F1 threshold
    optimal_f1_threshold = find_optimal_f1_threshold(all_predictions, all_labels)
    optimal_f1_metrics = evaluate_at_threshold(all_predictions, all_labels, optimal_f1_threshold)
    print(f"\nOptimal F1 Threshold ({optimal_f1_threshold:.4f}):")
    print_metrics_table(optimal_f1_metrics, "Metrics")
    
    # Operating threshold (prioritize recall, maintain specificity >= 0.93)
    operating_threshold = find_operating_threshold(all_predictions, all_labels, min_specificity=0.93)
    operating_metrics = evaluate_at_threshold(all_predictions, all_labels, operating_threshold)
    print(f"\nOperating Threshold ({operating_threshold:.4f}) - Prioritizes Recall, Specificity >= 0.93:")
    print_metrics_table(operating_metrics, "Metrics")
    print("\nThreshold Selection Rationale:")
    print(f"  - Operating threshold chosen to prioritize recall (sensitivity) while maintaining")
    print(f"    specificity >= 0.93 for clinical safety. This threshold balances the need to")
    print(f"    catch all positive cases (high recall) while minimizing false alarms (high specificity).")
    
    # 2. PROBABILITY CALIBRATION
    print("\n" + "="*60)
    print("2. PROBABILITY CALIBRATION")
    print("="*60)
    
    # Fit temperature scaling on validation set
    print("Fitting temperature scaling...")
    val_labels_path = os.path.join(config['data_dir'], 'val_labels.csv')
    val_images_dir = os.path.join(config['data_dir'], 'val')
    val_transform = get_transforms(config['img_size'], is_training=False)
    val_dataset = ChestXrayDataset(val_labels_path, val_images_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_logits.extend(logits.squeeze().cpu())
            val_labels.extend(labels.cpu().numpy())
    
    val_logits = torch.tensor(val_logits)
    val_labels = torch.tensor(val_labels)
    
    calibrator = TemperatureScaling()
    calibrator.fit(val_logits, val_labels)
    print(f"Temperature: {calibrator.temperature:.4f}")
    
    # Apply calibration to test predictions
    calibrated_logits = calibrator.transform(all_logits)
    calibrated_predictions = torch.sigmoid(calibrated_logits).numpy()
    
    # Compute ECE
    ece_uncalibrated = compute_ece(all_predictions, all_labels)
    ece_calibrated = compute_ece(calibrated_predictions, all_labels)
    
    print(f"\nExpected Calibration Error (ECE):")
    print(f"  Uncalibrated: {ece_uncalibrated:.4f}")
    print(f"  Calibrated: {ece_calibrated:.4f}")
    
    # Plot reliability diagram
    reliability_path = os.path.join(config['output_dir'], 'reliability_diagram.png')
    plot_reliability_diagram(calibrated_predictions, all_labels, save_path=reliability_path)
    
    # Evaluate with calibrated predictions
    calibrated_metrics = evaluate_at_threshold(calibrated_predictions, all_labels, default_threshold)
    print(f"\nCalibrated Predictions (threshold 0.5):")
    print_metrics_table(calibrated_metrics, "Metrics")
    
    # 3. ERROR ANALYSIS
    print("\n" + "="*60)
    print("3. ERROR ANALYSIS")
    print("="*60)
    
    error_analysis = analyze_errors(all_predictions, all_labels, default_threshold)
    print("\nError Analysis Summary:")
    for bullet in error_analysis['summary']:
        print(f"  • {bullet}")
    
    # 4. ROBUSTNESS CHECKS
    print("\n" + "="*60)
    print("4. ROBUSTNESS CHECKS")
    print("="*60)
    
    print("Evaluating with stronger augmentations...")
    # Get the test dataset from the loader
    test_dataset = test_loader.dataset
    robustness_results = evaluate_with_augmentations(
        model, test_dataset, device,
        config['img_size'], config['batch_size'], config['num_workers']
    )
    
    print("\nRobustness Test Results:")
    print(f"  Baseline AUROC: {default_metrics['auroc']:.4f}, F1: {default_metrics['f1']:.4f}, Recall: {default_metrics['recall']:.4f}")
    for aug_name, metrics in robustness_results.items():
        print(f"  {aug_name}: AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}, Recall: {metrics['recall']:.4f}")
    
    # 5. GENERATE FIGURES
    print("\n" + "="*60)
    print("5. GENERATING FIGURES")
    print("="*60)
    
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # ROC curve
    roc_path = os.path.join(config['output_dir'], 'roc_curve.png')
    plot_roc_curve(all_labels, all_predictions, save_path=roc_path, title="ROC Curve - Test Set")
    
    # Confusion matrices for each threshold
    for threshold_name, threshold, metrics in [
        ('default', default_threshold, default_metrics),
        ('optimal_f1', optimal_f1_threshold, optimal_f1_metrics),
        ('operating', operating_threshold, operating_metrics)
    ]:
        y_pred = (all_predictions >= threshold).astype(int)
        cm_path = os.path.join(config['output_dir'], f'confusion_matrix_{threshold_name}.png')
        plot_confusion_matrix(
            all_labels, y_pred, save_path=cm_path,
            title=f"Confusion Matrix - {threshold_name.replace('_', ' ').title()} Threshold ({threshold:.4f})"
        )
    
    # 6. CREATE METADATA
    print("\n" + "="*60)
    print("6. CREATING METADATA")
    print("="*60)
    
    metadata = create_metadata(config, model)
    metadata_path = os.path.join(config['output_dir'], 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    # 7. SAVE COMPREHENSIVE RESULTS
    print("\n" + "="*60)
    print("7. SAVING RESULTS")
    print("="*60)
    
    evaluation_results = {
        'thresholds': {
            'default': {
                'threshold': float(default_threshold),
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in default_metrics.items()},
                'explanation': 'Standard 0.5 threshold for binary classification'
            },
            'optimal_f1': {
                'threshold': float(optimal_f1_threshold),
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in optimal_f1_metrics.items()},
                'explanation': 'Threshold that maximizes F1 score'
            },
            'operating': {
                'threshold': float(operating_threshold),
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in operating_metrics.items()},
                'explanation': 'Operating threshold chosen to prioritize recall while maintaining specificity >= 0.93 for clinical safety'
            }
        },
        'calibration': {
            'temperature': float(calibrator.temperature),
            'ece_uncalibrated': float(ece_uncalibrated),
            'ece_calibrated': float(ece_calibrated),
            'calibrated_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                  for k, v in calibrated_metrics.items()}
        },
        'error_analysis': error_analysis,
        'robustness': robustness_results,
        'config': config
    }
    
    results_path = os.path.join(config['output_dir'], 'evaluation_results.json')
    save_metrics(evaluation_results, results_path)
    
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {results_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"All figures saved to: {config['output_dir']}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Default Threshold (0.5): AUROC={default_metrics['auroc']:.4f}, F1={default_metrics['f1']:.4f}")
    print(f"Optimal F1 Threshold ({optimal_f1_threshold:.4f}): F1={optimal_f1_metrics['f1']:.4f}")
    print(f"Operating Threshold ({operating_threshold:.4f}): Recall={operating_metrics['recall']:.4f}, Specificity={operating_metrics['specificity']:.4f}")
    print(f"ECE: Uncalibrated={ece_uncalibrated:.4f}, Calibrated={ece_calibrated:.4f}")


if __name__ == "__main__":
    main()

