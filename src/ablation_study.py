"""
Ablation study script to compare different model architectures.

This script trains and evaluates multiple model backbones (ResNet18, ResNet50, EfficientNetV2-S)
under identical conditions to compare performance, model size, and inference time.
"""

import os
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from config import get_config
from train import train_model
from eval_enhanced import main as eval_main
from model import create_model
from data import create_pre_split_data_loaders
from utils import get_device, seed_everything
from plotting import plot_precision_recall_curve, plot_roc_vs_threshold


def evaluate_model_performance(model, test_loader, device, n_runs=10):
    """
    Evaluate model inference performance.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        n_runs: Number of inference runs for timing
    
    Returns:
        dict: Performance metrics
    """
    model.eval()
    
    # Get a sample batch
    sample_batch = next(iter(test_loader))
    sample_images = sample_batch[0][:1].to(device)  # Single image
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_images)
    
    # Time inference
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    inference_times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.time()
            _ = model(sample_images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_times.append((time.time() - start) * 1000)  # ms
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'mean_inference_time_ms': np.mean(inference_times),
        'std_inference_time_ms': np.std(inference_times),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
    }


def run_ablation_study(models_to_test=None, data_dir=None, output_dir="./results/ablation"):
    """
    Run ablation study comparing different model architectures.
    
    Args:
        models_to_test: List of model names to test (default: ['resnet18', 'resnet50', 'efficientnet_v2_s'])
        data_dir: Path to data directory
        output_dir: Directory to save results
    """
    if models_to_test is None:
        models_to_test = ['resnet18', 'resnet50', 'efficientnet_v2_s']
    
    if data_dir is None:
        data_dir = "./data/chest_xray"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base config
    config = get_config()
    if data_dir:
        config['data_dir'] = data_dir
    
    seed_everything(config['seed'])
    device = get_device()
    
    results = {}
    
    print("=" * 80)
    print("ABLATION STUDY: Comparing Model Architectures")
    print("=" * 80)
    print(f"Models to test: {models_to_test}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load test data once
    print("Loading test data...")
    _, _, test_loader, _ = create_pre_split_data_loaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=0,  # Use 0 for compatibility
        use_weighted_sampling=False
    )
    
    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*80}")
        
        # Update config for this model
        model_config = config.copy()
        model_config['model_name'] = model_name
        model_config['output_dir'] = os.path.join(output_dir, model_name)
        model_config['best_model_path'] = os.path.join(model_config['output_dir'], 'best.pt')
        model_config['metrics_path'] = os.path.join(model_config['output_dir'], 'metrics.json')
        
        os.makedirs(model_config['output_dir'], exist_ok=True)
        
        try:
            # Train model
            print(f"\nTraining {model_name}...")
            model, history, final_metrics = train_model(model_config)
            
            # Evaluate on test set
            print(f"\nEvaluating {model_name} on test set...")
            model.eval()
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for images, labels in tqdm(test_loader, desc="Test Evaluation"):
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits.squeeze())
                    all_predictions.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            # Compute metrics
            from utils import compute_metrics
            test_metrics = compute_metrics(all_labels, all_predictions, threshold=0.5)
            
            # Performance evaluation
            print(f"\nEvaluating {model_name} performance...")
            perf_metrics = evaluate_model_performance(model, test_loader, device)
            
            # Save plots
            pr_path = os.path.join(model_config['output_dir'], 'precision_recall_curve.png')
            plot_precision_recall_curve(all_labels, all_predictions, save_path=pr_path)
            
            roc_thresh_path = os.path.join(model_config['output_dir'], 'roc_vs_threshold.png')
            plot_roc_vs_threshold(all_labels, all_predictions, save_path=roc_thresh_path)
            
            # Store results
            results[model_name] = {
                'test_metrics': test_metrics,
                'performance': perf_metrics,
                'best_val_auroc': final_metrics['best_val_auroc'],
                'training_epochs': len(history['train_loss']),
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            }
            
            print(f"\n{model_name} Results:")
            print(f"  Test AUROC: {test_metrics['auroc']:.4f}")
            print(f"  Test F1: {test_metrics['f1']:.4f}")
            print(f"  Parameters: {perf_metrics['total_parameters']:,}")
            print(f"  Model Size: {perf_metrics['model_size_mb']:.2f} MB")
            print(f"  Inference Time: {perf_metrics['mean_inference_time_ms']:.2f} ± {perf_metrics['std_inference_time_ms']:.2f} ms")
            
        except Exception as e:
            print(f"\nError testing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = {'error': str(e)}
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, 'ablation_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'AUROC':<10} {'F1':<10} {'Params (M)':<12} {'Size (MB)':<12} {'Inference (ms)':<15}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if 'error' not in result:
            metrics = result['test_metrics']
            perf = result['performance']
            print(f"{model_name:<20} {metrics['auroc']:<10.4f} {metrics['f1']:<10.4f} "
                  f"{perf['total_parameters']/1e6:<12.2f} {perf['model_size_mb']:<12.2f} "
                  f"{perf['mean_inference_time_ms']:<15.2f}")
        else:
            print(f"{model_name:<20} {'ERROR':<10}")
    
    print(f"\nDetailed results saved to: {comparison_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study comparing model architectures")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray",
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./results/ablation",
                       help="Directory to save results")
    parser.add_argument("--models", nargs="+", 
                       default=['resnet18', 'resnet50', 'efficientnet_v2_s'],
                       help="Model architectures to test")
    
    args = parser.parse_args()
    
    run_ablation_study(
        models_to_test=args.models,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

