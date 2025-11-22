"""
Prepare all figures for IEEE report.

This script copies all necessary images and generates any missing plots
for inclusion in the LaTeX report.
"""

import os
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from generate_additional_plots import generate_plots_from_eval_results


def prepare_ieee_figures():
    """Copy all required figures to docs/figs/ for IEEE report."""
    
    # Create docs/figs directory
    docs_figs_dir = Path("docs/figs")
    docs_figs_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path("results")
    
    # Mapping of source files to destination names
    figure_mapping = {
        # Core evaluation plots
        "roc_curve.png": "roc_curve.png",
        "confusion_matrix.png": "confusion_matrix.png",
        "loss_curve.png": "loss_curve.png",
        "precision_recall_curve.png": "pr_curve.png",
        "roc_vs_threshold.png": "threshold_curve.png",
        "f1_accuracy_vs_threshold.png": "f1_accuracy_vs_threshold.png",
        "calibration_curve.png": "calibration_curve.png",
        "reliability_diagram.png": "reliability_diagram.png",
        
        # Confusion matrices for different thresholds
        "confusion_matrix_default.png": "confusion_matrix_default.png",
        "confusion_matrix_optimal_f1.png": "confusion_matrix_optimal_f1.png",
        "confusion_matrix_operating.png": "confusion_matrix_operating.png",
        
        # Grad-CAM examples
        "cam_normal_001.png": "gradcam_normal_example.png",
        "cam_abnormal_001.png": "gradcam_abnormal_example.png",
        
        # Ablation study results
        "ablation/ablation_comparison.csv": "ablation_table.csv",
        "ablation/ablation_comparison.json": "ablation_table.json",
    }
    
    copied_files = []
    missing_files = []
    
    print("=" * 60)
    print("Preparing IEEE Report Figures")
    print("=" * 60)
    
    # Copy existing files
    for source_name, dest_name in figure_mapping.items():
        source_path = results_dir / source_name
        
        if source_path.exists():
            dest_path = docs_figs_dir / dest_name
            shutil.copy2(source_path, dest_path)
            copied_files.append(dest_name)
            print(f"✓ Copied: {source_name} -> {dest_name}")
        else:
            missing_files.append(source_name)
            print(f"✗ Missing: {source_name}")
    
    # Generate missing plots if predictions.npz exists
    predictions_path = results_dir / "predictions.npz"
    if predictions_path.exists():
        print("\nGenerating additional plots from saved predictions...")
        try:
            generate_plots_from_eval_results(
                predictions_path=str(predictions_path),
                output_dir=str(results_dir)
            )
            
            # Try copying again for newly generated plots
            for source_name, dest_name in figure_mapping.items():
                if source_name in missing_files:
                    source_path = results_dir / source_name
                    if source_path.exists():
                        dest_path = docs_figs_dir / dest_name
                        shutil.copy2(source_path, dest_path)
                        copied_files.append(dest_name)
                        missing_files.remove(source_name)
                        print(f"✓ Generated and copied: {source_name} -> {dest_name}")
        except Exception as e:
            print(f"Warning: Could not generate additional plots: {e}")
    
    # Create architecture diagram placeholder if missing
    arch_path = docs_figs_dir / "architecture.png"
    if not arch_path.exists():
        print("\n⚠ Note: architecture.png not found. Please create manually or use a diagram tool.")
        print("   Expected location: docs/figs/architecture.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✓ Successfully copied {len(copied_files)} files to docs/figs/")
    if missing_files:
        print(f"⚠ {len(missing_files)} files are missing:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nTo generate missing files, run:")
        print("  1. python src/train.py  # Generate loss_curve.png")
        print("  2. python src/eval_enhanced.py  # Generate evaluation plots")
        print("  3. python src/generate_additional_plots.py  # Generate additional plots")
        print("  4. python src/ablation_study.py  # Generate ablation results")
    
    print(f"\nAll figures are in: {docs_figs_dir.absolute()}")
    print("\nFor LaTeX inclusion, use:")
    print("  \\includegraphics[width=\\textwidth]{figs/roc_curve.png}")
    print("  \\includegraphics[width=\\textwidth]{figs/confusion_matrix.png}")
    print("  etc.")


if __name__ == "__main__":
    prepare_ieee_figures()


