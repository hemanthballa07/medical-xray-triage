"""
Training script for the Medical X-ray Triage project.

This script handles the complete training pipeline including data loading,
model training, validation, and checkpoint saving.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config, save_config
from data import create_data_loaders, get_simple_data_loader
from model import create_model, save_model
from utils import (
    seed_everything, compute_metrics, print_metrics_table, 
    save_metrics, get_device, format_time
)
import json
import time


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_score, model):
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        """Save the best model weights."""
        self.best_weights = model.state_dict()


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        dict: Training metrics for the epoch
    """
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        
        # Get predictions
        with torch.no_grad():
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # Compute epoch metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        dict: Validation metrics for the epoch
    """
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            logits = model(images)
            loss = criterion(logits.squeeze(), labels)
            
            # Statistics
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits.squeeze())
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
    
    # Compute epoch metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def train_model(config):
    """
    Main training function.
    
    Args:
        config: Configuration dictionary
    """
    start_time = time.time()
    
    # Set seed for reproducibility
    seed_everything(config['seed'])
    
    # Get device
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Save configuration
    config_path = os.path.join(config['output_dir'], 'params.yaml')
    save_config(config, config_path)
    
    # Create data loaders
    print("Loading data...")
    
    # For small datasets, use simple data loader
    if os.path.exists(config['labels_path']):
        import pandas as pd
        labels_df = pd.read_csv(config['labels_path'])
        if len(labels_df) <= 10:  # Small dataset, use simple loader
            print("Small dataset detected, using simple data loader")
            train_loader, dataset = get_simple_data_loader(
                labels_path=config['labels_path'],
                images_dir=config['images_dir'],
                batch_size=config['batch_size'],
                img_size=config['img_size'],
                is_training=True,
                num_workers=config['num_workers']
            )
            val_loader = train_loader  # Use same data for validation
            test_loader = train_loader  # Use same data for testing
            class_weights = {0: 1.0, 1: 1.0}  # Equal weights
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
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {class_weights}")
    
    # Create model
    print(f"Creating {config['model_name']} model...")
    model = create_model(
        model_name=config['model_name'],
        num_classes=1,
        pretrained=config['pretrained']
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function with class weights
    if len(class_weights) > 1:
        # Calculate class weights for loss function
        class_weights_tensor = torch.tensor([
            class_weights[0], class_weights[1]
        ], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1:2])
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Define optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Define scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=1
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_auroc = 0.0
    
    print("Starting training...")
    print("=" * 80)
    
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_metrics['auroc'])
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['train_acc'].append((train_metrics['precision'] * train_metrics['recall'] * 2) / 
                                   (train_metrics['precision'] + train_metrics['recall'] + 1e-8))
        history['val_acc'].append((val_metrics['precision'] * val_metrics['recall'] * 2) / 
                                 (val_metrics['precision'] + val_metrics['recall'] + 1e-8))
        
        # Print epoch results
        print(f"Train - Loss: {train_metrics['loss']:.4f}, AUROC: {train_metrics['auroc']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, AUROC: {val_metrics['auroc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            # Save state_dict only for smaller file size
            torch.save(model.state_dict(), config['best_model_path'])
            print(f"New best model saved! Val AUROC: {best_val_auroc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['auroc'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
        
        print()
    
    # Save final metrics and history
    final_metrics = {
        'best_val_auroc': best_val_auroc,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics,
        'history': history,
        'config': config
    }
    
    save_metrics(final_metrics, config['metrics_path'])
    
    print("Training completed!")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    
    # Save final metrics with timestamp
    final_metrics['training_time'] = time.time() - start_time
    final_metrics['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_metrics(final_metrics, config['metrics_path'])
    
    # Print one-line summary
    print(f"SUMMARY: Best Val AUROC: {best_val_auroc:.4f}, Epochs: {len(history['train_loss'])}, Time: {format_time(time.time() - start_time)}")
    
    return model, history, final_metrics


def main():
    """Main function."""
    # Get configuration
    config = get_config()
    
    print("Medical X-ray Triage Training")
    print("=" * 50)
    print(f"Model: {config['model_name']}")
    print(f"Data: {config['data_dir']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Device: {config['device']}")
    print()
    
    # Check if sample data exists
    if not os.path.exists(config['labels_path']):
        print("Sample data not found. Generating sample data...")
        from make_sample_data import create_sample_dataset
        create_sample_dataset(config['data_dir'])
        print()
    
    # Train model
    model, history, final_metrics = train_model(config)
    
    print("\nTraining Summary:")
    print_metrics_table(final_metrics['final_val_metrics'], "Final Validation Metrics")


if __name__ == "__main__":
    main()
