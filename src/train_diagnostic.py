"""
Diagnostic training script with comprehensive checks to verify training pipeline.

This script adds extensive debugging and verification to ensure:
- Dataset sizes are correct
- Batch shapes are correct
- Weights are updating
- Model modes are set correctly
- Augmentations are applied
- Training can overfit small subset
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
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config, save_config
from data import create_data_loaders, get_simple_data_loader, create_pre_split_data_loaders, ChestXrayDataset
from model import create_model, save_model
from utils import (
    seed_everything, compute_metrics, print_metrics_table, 
    save_metrics, get_device, format_time
)
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


def verify_augmentations(dataset, num_samples=5, save_dir="./results"):
    """Verify that augmentations are being applied."""
    print("\n" + "="*60)
    print("VERIFYING AUGMENTATIONS")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize augmented samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Show multiple augmented samples from same index
    for i in range(num_samples):
        aug_image, aug_label = dataset[0]  # Same index, different augmentation
        if isinstance(aug_image, torch.Tensor):
            # Denormalize for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            aug_image = aug_image * std + mean
            aug_image = torch.clamp(aug_image, 0, 1)
            aug_image = transforms.ToPILImage()(aug_image)
        
        axes[i].imshow(aug_image, cmap='gray' if aug_image.mode == 'L' else None)
        axes[i].set_title(f"Aug {i+1} (Label: {aug_label})")
        axes[i].axis('off')
    
    plt.tight_layout()
    aug_path = os.path.join(save_dir, 'augmentation_verification.png')
    plt.savefig(aug_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Augmentation samples saved to: {aug_path}")
    print("  Check the images to verify augmentations are being applied.")
    print("  (If images look identical, augmentations may not be working)")


def check_weight_updates(model, optimizer, train_loader, criterion, device):
    """Verify that weights are actually updating during training."""
    print("\n" + "="*60)
    print("CHECKING WEIGHT UPDATES")
    print("="*60)
    
    # Get a reference to a specific parameter
    # For ResNet18, check the first conv layer or classifier
    param_to_check = None
    param_name = None
    
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'conv1'):
        if hasattr(model.backbone.conv1, 'weight'):
            param_to_check = model.backbone.conv1.weight
            param_name = "backbone.conv1.weight"
    elif hasattr(model, 'classifier'):
        if hasattr(model.classifier[0], 'weight'):
            param_to_check = model.classifier[0].weight
            param_name = "classifier[0].weight"
    
    # Fallback to first parameter
    if param_to_check is None:
        param_to_check = next(model.parameters())
        param_name = "first_parameter"
    
    # Get initial weight
    old_param = param_to_check.clone().detach()
    
    # Run one training step
    model.train()
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits.squeeze(), labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Get new weight
    new_param = param_to_check
    
    # Calculate change
    param_change = (old_param - new_param).abs().mean().item()
    
    print(f"Parameter checked: {param_name}")
    print(f"Parameter shape: {old_param.shape}")
    print(f"Mean absolute weight change: {param_change:.8f}")
    
    if param_change < 1e-8:
        print("⚠ WARNING: Weights are NOT updating! Training pipeline may be broken.")
        return False
    else:
        print("✓ Weights are updating correctly.")
        return True


def overfit_test(model, train_dataset, device, num_epochs=50):
    """Test if model can overfit a tiny subset (sanity check)."""
    print("\n" + "="*60)
    print("OVERFIT TEST (Sanity Check)")
    print("="*60)
    
    # Create tiny subset
    subset_size = min(100, len(train_dataset))
    subset_indices = list(range(subset_size))
    subset = torch.utils.data.Subset(train_dataset, subset_indices)
    subset_loader = torch.utils.data.DataLoader(
        subset, batch_size=32, shuffle=True, num_workers=0
    )
    
    print(f"Testing on {subset_size} samples for {num_epochs} epochs...")
    
    # Create fresh model
    model_copy = create_model('resnet18', num_classes=1, pretrained=True)
    model_copy.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model_copy.parameters(), lr=0.001)
    
    model_copy.train()
    losses = []
    accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in subset_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model_copy(images)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate accuracy
            probs = torch.sigmoid(logits.squeeze())
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = epoch_loss / len(subset_loader)
        accuracy = correct / total
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
    
    final_loss = losses[-1]
    final_acc = accuracies[-1]
    
    print(f"\nFinal Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")
    
    if final_acc > 0.95 and final_loss < 0.1:
        print("✓ Model CAN overfit tiny subset - training pipeline is working!")
        return True
    else:
        print("⚠ WARNING: Model CANNOT overfit tiny subset - training pipeline may be broken!")
        return False


def train_epoch(model, train_loader, criterion, optimizer, device, epoch=0, debug=False):
    """Train the model for one epoch with diagnostic checks."""
    model.train()  # Ensure training mode
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        # DIAGNOSTIC: Check batch shapes on first batch
        if batch_idx == 0 and debug:
            print(f"\n[DEBUG] First batch shapes:")
            print(f"  Images shape: {images.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Image dtype: {images.dtype}, Label dtype: {labels.dtype}")
            print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"  Label range: [{labels.min():.1f}, {labels.max():.1f}]")
            print(f"  Model training mode: {model.training}")
        
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        # Gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    """Validate the model for one epoch."""
    model.eval()  # Ensure evaluation mode
    
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


def train_model(config, run_diagnostics=True):
    """Main training function with comprehensive diagnostics."""
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
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Check if this is a pre-split NIH dataset
    train_labels_path = os.path.join(config['data_dir'], 'train_labels.csv')
    is_nih_dataset = os.path.exists(train_labels_path)
    
    if is_nih_dataset:
        print("NIH dataset detected (pre-split structure), using pre-split data loaders")
        train_loader, val_loader, test_loader, class_weights = create_pre_split_data_loaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers']
        )
        
        # Get actual dataset sizes
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        print(f"\n✓ Dataset Sizes:")
        print(f"  Train dataset size: {len(train_dataset)}")
        print(f"  Validation dataset size: {len(val_dataset)}")
        print(f"  Train loader batches per epoch: {len(train_loader)}")
        print(f"  Validation loader batches per epoch: {len(val_loader)}")
        print(f"  Samples per batch: {config['batch_size']}")
        print(f"  Expected train samples: {len(train_loader) * config['batch_size']} (may vary with weighted sampler)")
        
    elif os.path.exists(config['labels_path']):
        import pandas as pd
        labels_df = pd.read_csv(config['labels_path'])
        if len(labels_df) <= 10:
            print("Small dataset detected, using simple data loader")
            train_loader, dataset = get_simple_data_loader(
                labels_path=config['labels_path'],
                images_dir=config['images_dir'],
                batch_size=config['batch_size'],
                img_size=config['img_size'],
                is_training=True,
                num_workers=config['num_workers']
            )
            val_loader = train_loader
            test_loader = train_loader
            class_weights = {0: 1.0, 1: 1.0}
            train_dataset = dataset
            val_dataset = dataset
        else:
            train_loader, val_loader, test_loader, class_weights = create_data_loaders(
                labels_path=config['labels_path'],
                images_dir=config['images_dir'],
                batch_size=config['batch_size'],
                img_size=config['img_size'],
                num_workers=config['num_workers']
            )
            train_dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset
            val_dataset = val_loader.dataset.dataset if hasattr(val_loader.dataset, 'dataset') else val_loader.dataset
    else:
        raise FileNotFoundError(f"Labels file not found: {config['labels_path']}")
    
    print(f"\n✓ Loader Information:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Class weights: {class_weights}")
    
    # Verify dataset is not empty
    if len(train_dataset) == 0:
        raise ValueError("ERROR: Train dataset is empty!")
    if len(val_dataset) == 0:
        raise ValueError("ERROR: Validation dataset is empty!")
    
    print(f"\n✓ Input Size:")
    print(f"  Image size: {config['img_size']}x{config['img_size']}")
    
    # Verify augmentations
    if run_diagnostics and is_nih_dataset:
        verify_augmentations(train_dataset, num_samples=5, save_dir=config['output_dir'])
    
    # Create model
    print(f"\n" + "="*60)
    print("CREATING MODEL")
    print("="*60)
    print(f"Model: {config['model_name']}")
    model = create_model(
        model_name=config['model_name'],
        num_classes=1,
        pretrained=config['pretrained']
    )
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Define loss function with class weights
    if len(class_weights) > 1:
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
        patience=3,
        min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
    # Run diagnostics
    if run_diagnostics:
        # Check weight updates
        weight_updates_ok = check_weight_updates(model, optimizer, train_loader, criterion, device)
        
        # Overfit test
        if is_nih_dataset:
            overfit_ok = overfit_test(model, train_dataset, device, num_epochs=50)
        else:
            print("\n⚠ Skipping overfit test (requires full dataset)")
            overfit_ok = None
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_auroc': [], 'val_auroc': [],
        'train_f1': [], 'val_f1': [],
        'train_acc': [], 'val_acc': []
    }
    
    best_val_auroc = 0.0
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 40)
        
        # Train with debug on first epoch
        debug = (epoch == 0 and run_diagnostics)
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, debug=debug)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Verify model modes
        if run_diagnostics and epoch == 0:
            print(f"\n[DEBUG] Model mode verification:")
            print(f"  After training: model.training = {model.training}")
            model.eval()
            print(f"  After model.eval(): model.training = {model.training}")
            model.train()
        
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
        
        # Check loss curve behavior
        if epoch > 0 and run_diagnostics:
            loss_change = history['train_loss'][-2] - history['train_loss'][-1]
            if loss_change < 0 and abs(loss_change) < 1e-6:
                print(f"⚠ WARNING: Loss barely changed ({loss_change:.8f})")
        
        # Save best model
        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), config['best_model_path'])
            print(f"✓ New best model saved! Val AUROC: {best_val_auroc:.4f}")
        
        # Early stopping
        if early_stopping(val_metrics['auroc'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final metrics and history
    final_metrics = {
        'best_val_auroc': best_val_auroc,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics,
        'history': history,
        'config': config
    }
    
    save_metrics(final_metrics, config['metrics_path'])
    
    # Print diagnostic summary
    if run_diagnostics:
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        print(f"✓ Dataset size: Train={len(train_dataset)}, Val={len(val_dataset)}")
        print(f"✓ Input resolution: {config['img_size']}x{config['img_size']}")
        print(f"✓ Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        if 'weight_updates_ok' in locals():
            print(f"{'✓' if weight_updates_ok else '⚠'} Weight updates: {'OK' if weight_updates_ok else 'FAILED'}")
        if 'overfit_ok' in locals() and overfit_ok is not None:
            print(f"{'✓' if overfit_ok else '⚠'} Overfit test: {'PASSED' if overfit_ok else 'FAILED'}")
        
        # Loss curve analysis
        if len(history['train_loss']) > 1:
            initial_loss = history['train_loss'][0]
            final_loss = history['train_loss'][-1]
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            print(f"✓ Loss reduction: {initial_loss:.4f} → {final_loss:.4f} ({loss_reduction:.1f}% reduction)")
            
            if loss_reduction < 5:
                print("⚠ WARNING: Loss barely decreased - training may not be working properly")
        
        # Training speed analysis
        training_time = time.time() - start_time
        samples_per_second = (len(train_dataset) * len(history['train_loss'])) / training_time
        print(f"✓ Training speed: {samples_per_second:.1f} samples/second")
        print(f"✓ Total training time: {format_time(training_time)}")
        
        # Reasonable speed check
        if config['img_size'] == 320 and total_params < 15_000_000:
            print(f"✓ Fast training is expected for ResNet18 at 320x320 resolution")
    
    print("\nTraining completed!")
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
    
    print("Medical X-ray Triage Training (Diagnostic Mode)")
    print("=" * 60)
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
    
    # Train model with diagnostics
    model, history, final_metrics = train_model(config, run_diagnostics=True)
    
    print("\nTraining Summary:")
    print_metrics_table(final_metrics['final_val_metrics'], "Final Validation Metrics")


if __name__ == "__main__":
    main()

