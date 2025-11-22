"""
Automated hyperparameter sweeps using Optuna.

This module provides functionality to automatically search for optimal
hyperparameters including learning rate, weight decay, and augmentation intensity.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(str(Path(__file__).parent))

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Install with: pip install optuna")

from config import get_config
from data import create_pre_split_data_loaders
from model import create_model
from train import train_epoch, evaluate
from utils import get_device, seed_everything, save_metrics


def objective(trial, data_dir, device, img_size=320, epochs=5, quick_mode=True):
    """
    Objective function for Optuna optimization.
    
    Args:
        trial: Optuna trial object
        data_dir: Data directory
        device: Device to train on
        img_size: Image size
        epochs: Number of epochs per trial
        quick_mode: If True, use fewer epochs for faster search
    
    Returns:
        float: Validation AUROC (to maximize)
    """
    # Suggest hyperparameters
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    
    # Augmentation intensity
    rotation_degrees = trial.suggest_int('rotation_degrees', 5, 30)
    color_jitter_brightness = trial.suggest_float('color_jitter_brightness', 0.0, 0.3)
    color_jitter_contrast = trial.suggest_float('color_jitter_contrast', 0.0, 0.3)
    
    # Create data loaders with custom augmentation
    try:
        train_loader, val_loader, _, class_weights = create_pre_split_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            num_workers=0,  # Avoid multiprocessing issues
            use_weighted_sampling=True
        )
        
        # Create model
        model = create_model(model_name='resnet18', num_classes=1, pretrained=True)
        model.to(device)
        
        # Optimizer with suggested hyperparameters
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Training loop
        best_val_auroc = 0.0
        
        for epoch in range(epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, class_weights
            )
            
            # Validate
            val_metrics = evaluate(model, val_loader, device, class_weights)
            
            # Track best validation AUROC
            if val_metrics['auroc'] > best_val_auroc:
                best_val_auroc = val_metrics['auroc']
            
            # Report intermediate value for pruning
            trial.report(val_metrics['auroc'], epoch)
            
            # Prune if trial is not promising
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return best_val_auroc
    
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Return low score for failed trials


def run_hyperparameter_sweep(
    data_dir: str,
    n_trials: int = 20,
    epochs_per_trial: int = 5,
    output_dir: str = "./results/hyperparameter_sweep",
    study_name: str = "pneumonia_classification"
):
    """
    Run hyperparameter sweep using Optuna.
    
    Args:
        data_dir: Data directory
        n_trials: Number of trials to run
        epochs_per_trial: Number of epochs per trial
        output_dir: Directory to save results
        study_name: Name of the Optuna study
    """
    if not OPTUNA_AVAILABLE:
        print("Error: Optuna is not installed. Install with: pip install optuna")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()
    seed_everything(42)
    
    print(f"Starting hyperparameter sweep with {n_trials} trials...")
    print(f"Each trial will train for {epochs_per_trial} epochs")
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, data_dir, device, epochs=epochs_per_trial),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("HYPERPARAMETER SWEEP RESULTS")
    print("="*60)
    print(f"\nBest trial:")
    print(f"  Value (Val AUROC): {study.best_value:.4f}")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_value': float(study.best_value),
        'best_params': study.best_params,
        'n_trials': n_trials,
        'trials_data': []
    }
    
    for trial in study.trials:
        results['trials_data'].append({
            'number': trial.number,
            'value': float(trial.value) if trial.value is not None else None,
            'params': trial.params,
            'state': str(trial.state)
        })
    
    results_path = os.path.join(output_dir, 'hyperparameter_sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Generate visualizations
    try:
        # Optimization history
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        # Parameter importance
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, 'param_importances.png'))
        except:
            pass  # May fail if not enough trials
        
        # Parallel coordinate plot
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        except:
            pass
        
        print("Visualizations saved to:", output_dir)
    except Exception as e:
        print(f"Warning: Could not generate all visualizations: {e}")
    
    # Create 3D/contour plots for key parameters
    try:
        create_metric_surfaces(study, output_dir)
    except Exception as e:
        print(f"Warning: Could not create metric surfaces: {e}")
    
    return study


def create_metric_surfaces(study, output_dir):
    """
    Create 3D and contour plots showing metric surfaces.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save plots
    """
    if len(study.trials) < 5:
        print("Not enough trials for surface plots")
        return
    
    # Extract data
    trials_data = []
    for trial in study.trials:
        if trial.value is not None and 'learning_rate' in trial.params and 'weight_decay' in trial.params:
            trials_data.append({
                'lr': trial.params['learning_rate'],
                'wd': trial.params['weight_decay'],
                'value': trial.value
            })
    
    if len(trials_data) < 5:
        return
    
    lrs = [t['lr'] for t in trials_data]
    wds = [t['wd'] for t in trials_data]
    values = [t['value'] for t in trials_data]
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(lrs, wds, values, c=values, cmap='viridis', s=50)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('Weight Decay')
    ax1.set_zlabel('Validation AUROC')
    ax1.set_title('Hyperparameter Search Space (3D)')
    plt.colorbar(scatter, ax=ax1, label='AUROC')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    # Create grid for contour
    from scipy.interpolate import griddata
    import numpy as np
    
    lr_grid = np.logspace(np.log10(min(lrs)), np.log10(max(lrs)), 50)
    wd_grid = np.logspace(np.log10(min(wds)), np.log10(max(wds)), 50)
    LR, WD = np.meshgrid(lr_grid, wd_grid)
    
    # Interpolate values
    Z = griddata((lrs, wds), values, (LR, WD), method='cubic', fill_value=np.nan)
    
    contour = ax2.contourf(LR, WD, Z, levels=20, cmap='viridis')
    ax2.scatter(lrs, wds, c=values, cmap='viridis', s=30, edgecolors='black', linewidths=0.5)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Weight Decay')
    ax2.set_title('Metric Surface (Contour)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.colorbar(contour, ax=ax2, label='AUROC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metric_surfaces.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Metric surfaces plot saved")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray",
                       help="Data directory")
    parser.add_argument("--n_trials", type=int, default=20,
                       help="Number of trials")
    parser.add_argument("--epochs_per_trial", type=int, default=5,
                       help="Epochs per trial")
    parser.add_argument("--output_dir", type=str, default="./results/hyperparameter_sweep",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not OPTUNA_AVAILABLE:
        print("Please install Optuna: pip install optuna")
        sys.exit(1)
    
    run_hyperparameter_sweep(
        args.data_dir, args.n_trials, args.epochs_per_trial, args.output_dir
    )

