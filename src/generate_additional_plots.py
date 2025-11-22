"""
Generate additional evaluation plots from existing evaluation results.

This script creates precision-recall curves and ROC vs threshold plots
from evaluation_results.json.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from plotting import (
    plot_precision_recall_curve, 
    plot_roc_vs_threshold,
    plot_f1_accuracy_vs_threshold,
    plot_calibration_curve
)


def generate_plots_from_eval_results(predictions_path="./results/predictions.npz",
                                     output_dir="./results"):
    """
    Generate precision-recall and ROC vs threshold plots from saved predictions.
    
    Args:
        predictions_path: Path to predictions.npz file
        output_dir: Directory to save plots
    """
    # Try to load saved predictions first
    if os.path.exists(predictions_path):
        print(f"Loading predictions from {predictions_path}...")
        data = np.load(predictions_path)
        all_predictions = data['predictions']
        all_labels = data['labels']
    else:
        print(f"Error: Predictions file not found at {predictions_path}")
        print("Please run evaluation first: python src/eval_enhanced.py")
        print("This will create predictions.npz automatically.")
        return
    
    # Generate plots
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating precision-recall curve...")
    pr_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plot_precision_recall_curve(all_labels, all_predictions, save_path=pr_path,
                                title="Precision-Recall Curve - Test Set")
    
    print("Generating ROC vs threshold plot...")
    roc_thresh_path = os.path.join(output_dir, 'roc_vs_threshold.png')
    plot_roc_vs_threshold(all_labels, all_predictions, save_path=roc_thresh_path,
                          title="ROC Metrics vs Threshold - Test Set")
    
    print("Generating F1/Accuracy vs threshold plot...")
    f1_acc_path = os.path.join(output_dir, 'f1_accuracy_vs_threshold.png')
    plot_f1_accuracy_vs_threshold(all_labels, all_predictions, save_path=f1_acc_path,
                                  title="F1 and Accuracy vs Threshold - Test Set")
    
    print("Generating calibration curve...")
    calib_path = os.path.join(output_dir, 'calibration_curve.png')
    plot_calibration_curve(all_labels, all_predictions, save_path=calib_path,
                          title="Calibration Curve - Test Set")
    
    print(f"\nAll plots saved to:")
    print(f"  - {pr_path}")
    print(f"  - {roc_thresh_path}")
    print(f"  - {f1_acc_path}")
    print(f"  - {calib_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate additional evaluation plots")
    parser.add_argument("--predictions", type=str, default="./results/predictions.npz",
                       help="Path to predictions.npz file")
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    generate_plots_from_eval_results(args.predictions, args.output_dir)

