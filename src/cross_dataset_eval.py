"""
Cross-dataset evaluation for model generalization.

This module provides functionality to evaluate trained models on different
datasets to assess generalization and external validity.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent))

from config import get_config
from data import create_pre_split_data_loaders, get_transforms
from model import create_model, load_model
from utils import get_device, seed_everything, compute_metrics, print_metrics_table
from bootstrap_metrics import bootstrap_all_metrics


def evaluate_on_dataset(
    model_path: str,
    data_dir: str,
    dataset_name: str,
    device: torch.device,
    batch_size: int = 8,
    img_size: int = 320
) -> dict:
    """
    Evaluate model on a specific dataset.
    
    Args:
        model_path: Path to trained model
        data_dir: Directory containing the dataset
        dataset_name: Name of the dataset (for reporting)
        device: Device to run evaluation on
        batch_size: Batch size for evaluation
        img_size: Image size
    
    Returns:
        dict: Evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name}")
    print(f"{'='*60}")
    
    # Load model
    model, checkpoint_info = load_model(model_path, device)
    model.eval()
    
    # Create data loaders
    try:
        _, _, test_loader, _ = create_pre_split_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=0,  # Avoid multiprocessing issues
            use_weighted_sampling=False
        )
        print(f"Test dataset size: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Evaluate
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    from utils import compute_metrics
    metrics = compute_metrics(all_labels, all_predictions, threshold=0.5)
    
    # Bootstrap confidence intervals
    print(f"\nComputing bootstrap confidence intervals for {dataset_name}...")
    bootstrap_results = bootstrap_all_metrics(
        all_labels, all_predictions, threshold=0.5,
        n_bootstrap=500, confidence_level=0.95  # Fewer samples for speed
    )
    
    results = {
        'dataset_name': dataset_name,
        'dataset_size': len(all_labels),
        'metrics': metrics,
        'bootstrap': bootstrap_results
    }
    
    print_metrics_table(metrics, f"{dataset_name} Performance")
    print(f"\nBootstrap 95% CI:")
    for metric_name, metric_data in bootstrap_results.items():
        print(f"  {metric_name.upper()}: {metric_data['ci']}")
    
    return results


def cross_dataset_evaluation(
    model_path: str,
    datasets: list,
    output_dir: str = "./results",
    batch_size: int = 8,
    img_size: int = 320
):
    """
    Evaluate model on multiple datasets.
    
    Args:
        model_path: Path to trained model
        datasets: List of (dataset_name, data_dir) tuples
        output_dir: Directory to save results
        batch_size: Batch size for evaluation
        img_size: Image size
    """
    device = get_device()
    seed_everything(42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for dataset_name, data_dir in datasets:
        if not os.path.exists(data_dir):
            print(f"⚠️  Dataset {dataset_name} not found at {data_dir}, skipping...")
            continue
        
        results = evaluate_on_dataset(
            model_path, data_dir, dataset_name, device, batch_size, img_size
        )
        
        if results:
            all_results[dataset_name] = results
    
    # Save results
    results_path = os.path.join(output_dir, 'cross_dataset_evaluation.json')
    
    # Convert numpy types to native Python types for JSON
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    with open(results_path, 'w') as f:
        json.dump(convert_to_native(all_results), f, indent=2)
    
    print(f"\n{'='*60}")
    print("Cross-Dataset Evaluation Summary")
    print(f"{'='*60}")
    
    for dataset_name, results in all_results.items():
        metrics = results['metrics']
        print(f"\n{dataset_name}:")
        print(f"  AUROC: {metrics['auroc']:.4f} {results['bootstrap']['auroc']['ci']}")
        print(f"  F1: {metrics['f1']:.4f} {results['bootstrap']['f1']['ci']}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--model_path", type=str, default="./results/best.pt",
                       help="Path to trained model")
    parser.add_argument("--datasets", nargs='+', 
                       default=[("ChestXray", "./data/chest_xray")],
                       help="List of (name, path) dataset tuples")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Parse datasets (format: name1:path1 name2:path2)
    datasets = []
    for dataset_str in args.datasets:
        if ':' in dataset_str:
            name, path = dataset_str.split(':', 1)
            datasets.append((name, path))
        else:
            datasets.append((dataset_str, dataset_str))
    
    cross_dataset_evaluation(
        args.model_path, datasets, args.output_dir
    )

