#!/usr/bin/env python3
"""Generate loss curve from training history."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_loss_curve(metrics_path=None, output_path=None):
    """Plot training and validation loss curves."""
    
    # Set default paths relative to project root
    if metrics_path is None:
        project_root = Path(__file__).parent.parent
        metrics_path = project_root / "results/metrics.json"
    if output_path is None:
        project_root = Path(__file__).parent.parent
        output_path = project_root / "results/loss_curve.png"
    
    metrics_path = str(metrics_path)
    output_path = str(output_path)
    
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    history = metrics['history']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    # Get number of epochs
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot loss curves
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=8)
    plt.plot(epochs, val_loss, 'r-s', label='Validation Loss', linewidth=2, markersize=8)
    
    # Customize plot
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curve saved to: {output_path}")
    print(f"Training epochs: {len(train_loss)}")
    print(f"Final training loss: {train_loss[-1]:.4f}")
    print(f"Final validation loss: {val_loss[-1]:.4f}")


if __name__ == "__main__":
    plot_loss_curve()

