"""
Create a pipeline flow diagram showing how results flow into the UI.

This script generates a visual diagram showing the complete ML pipeline
from data ingestion to UI deployment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_pipeline_diagram(save_path="docs/pipeline_flow.png"):
    """
    Create a pipeline flow diagram.
    
    Args:
        save_path: Path to save the diagram
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    data_color = '#4A90E2'
    process_color = '#50C878'
    model_color = '#FF6B6B'
    ui_color = '#FFD93D'
    result_color = '#9B59B6'
    
    # Title
    ax.text(8, 9.5, 'Medical X-ray Triage System Pipeline', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    
    # Stage 1: Data Ingestion
    data_box = FancyBboxPatch((0.5, 7), 2.5, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.75, 8, 'Data\nIngestion', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    ax.text(1.75, 7.3, 'Chest X-ray\nImages', ha='center', va='center', 
            fontsize=10, color='white')
    
    # Stage 2: Preprocessing
    preprocess_box = FancyBboxPatch((3.5, 7), 2.5, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(preprocess_box)
    ax.text(4.75, 8, 'Preprocessing', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(4.75, 7.3, 'Resize, Normalize\nAugmentation', ha='center', va='center',
            fontsize=10, color='white')
    
    # Stage 3: Training
    train_box = FancyBboxPatch((6.5, 7), 2.5, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=model_color, edgecolor='black', linewidth=2)
    ax.add_patch(train_box)
    ax.text(7.75, 8, 'Model\nTraining', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(7.75, 7.3, 'ResNet18\nTransfer Learning', ha='center', va='center',
            fontsize=10, color='white')
    
    # Stage 4: Evaluation
    eval_box = FancyBboxPatch((9.5, 7), 2.5, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(eval_box)
    ax.text(10.75, 8, 'Evaluation', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(10.75, 7.3, 'Metrics, Plots\nCalibration', ha='center', va='center',
            fontsize=10, color='white')
    
    # Stage 5: Results Storage
    results_box = FancyBboxPatch((12.5, 7), 2.5, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=result_color, edgecolor='black', linewidth=2)
    ax.add_patch(results_box)
    ax.text(13.75, 8, 'Results\nStorage', ha='center', va='center',
            fontsize=12, fontweight='bold', color='white')
    ax.text(13.75, 7.3, 'best.pt\nmetrics.json', ha='center', va='center',
            fontsize=10, color='white')
    
    # Arrows between stages
    arrows = [
        ((3, 7.75), (3.5, 7.75)),  # Data -> Preprocess
        ((6, 7.75), (6.5, 7.75)),  # Preprocess -> Train
        ((9, 7.75), (9.5, 7.75)),  # Train -> Eval
        ((12, 7.75), (12.5, 7.75)), # Eval -> Results
    ]
    
    for (x1, y1), (x2, y2) in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle='->', lw=2, color='black')
        ax.add_patch(arrow)
    
    # Results to UI (downward flow)
    results_to_ui_arrow = FancyArrowPatch((13.75, 7), (13.75, 5.5),
                                         arrowstyle='->', lw=2, color='black')
    ax.add_patch(results_to_ui_arrow)
    
    # UI Box
    ui_box = FancyBboxPatch((10, 4), 7.5, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=ui_color, edgecolor='black', linewidth=2)
    ax.add_patch(ui_box)
    ax.text(13.75, 5, 'Streamlit UI', ha='center', va='center',
            fontsize=14, fontweight='bold', color='black')
    ax.text(13.75, 4.4, 'Upload • Predict • Visualize • Explain', ha='center', va='center',
            fontsize=11, color='black')
    
    # UI Components
    ui_components = [
        ('Image Upload', 11, 3.2),
        ('Grad-CAM', 13.75, 3.2),
        ('Uncertainty', 16.5, 3.2),
    ]
    
    for comp_name, x, y in ui_components:
        comp_box = FancyBboxPatch((x-0.8, y), 1.6, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='white', edgecolor=ui_color, linewidth=1.5)
        ax.add_patch(comp_box)
        ax.text(x, y+0.3, comp_name, ha='center', va='center',
                fontsize=9, color='black')
    
    # Results details (left side)
    results_details = [
        ('best.pt', 0.5, 5),
        ('metrics.json', 0.5, 4.5),
        ('roc_curve.png', 0.5, 4),
        ('predictions.npz', 0.5, 3.5),
    ]
    
    for detail, x, y in results_details:
        detail_box = FancyBboxPatch((x-0.2, y-0.15), 2.4, 0.3,
                                   boxstyle="round,pad=0.05",
                                   facecolor=result_color, edgecolor='black', linewidth=1,
                                   alpha=0.7)
        ax.add_patch(detail_box)
        ax.text(x+1, y, detail, ha='left', va='center',
                fontsize=9, color='white')
    
    # Arrow from results details to UI
    detail_arrow = FancyArrowPatch((3, 4.25), (10, 4.25),
                                   arrowstyle='->', lw=1.5, color='gray', linestyle='--')
    ax.add_patch(detail_arrow)
    ax.text(6.5, 4.5, 'Loaded by UI', ha='center', va='bottom',
            fontsize=9, color='gray', style='italic')
    
    # Evaluation outputs (right side)
    eval_outputs = [
        ('AUROC, F1', 13.5, 5.5),
        ('Confusion Matrix', 13.5, 5),
        ('Bootstrap CI', 13.5, 4.5),
        ('Failure Cases', 13.5, 4),
    ]
    
    for output, x, y in eval_outputs:
        output_box = FancyBboxPatch((x-0.2, y-0.15), 2.4, 0.3,
                                    boxstyle="round,pad=0.05",
                                    facecolor=process_color, edgecolor='black', linewidth=1,
                                    alpha=0.7)
        ax.add_patch(output_box)
        ax.text(x+1, y, output, ha='left', va='center',
                fontsize=9, color='white')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=data_color, label='Data'),
        mpatches.Patch(facecolor=process_color, label='Processing'),
        mpatches.Patch(facecolor=model_color, label='Model'),
        mpatches.Patch(facecolor=result_color, label='Results'),
        mpatches.Patch(facecolor=ui_color, label='UI'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Pipeline flow diagram saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs("docs", exist_ok=True)
    create_pipeline_diagram()

