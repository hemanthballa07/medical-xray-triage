"""
Generate documentation diagrams for the Medical X-ray Triage project.

This script creates architecture diagrams and UI wireframes for the documentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os


def create_architecture_diagram():
    """Create system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Medical X-ray Triage System Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Data Layer
    data_box = FancyBboxPatch((0.5, 9.5), 2, 1.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='blue')
    ax.add_patch(data_box)
    ax.text(1.5, 10.25, 'Data Layer', ha='center', va='center', fontweight='bold')
    ax.text(1.5, 9.8, '• Chest X-ray Images\n• Labels (Normal/Abnormal)\n• Data Augmentation', 
            ha='center', va='center', fontsize=10)
    
    # Model Layer
    model_box = FancyBboxPatch((3.5, 9.5), 2, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightgreen', edgecolor='green')
    ax.add_patch(model_box)
    ax.text(4.5, 10.25, 'Model Layer', ha='center', va='center', fontweight='bold')
    ax.text(4.5, 9.8, '• ResNet50/EfficientNet\n• Pretrained Backbone\n• Custom Classifier Head', 
            ha='center', va='center', fontsize=10)
    
    # Training Layer
    train_box = FancyBboxPatch((6.5, 9.5), 2, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='lightyellow', edgecolor='orange')
    ax.add_patch(train_box)
    ax.text(7.5, 10.25, 'Training Layer', ha='center', va='center', fontweight='bold')
    ax.text(7.5, 9.8, '• BCE Loss\n• Adam Optimizer\n• Early Stopping', 
            ha='center', va='center', fontsize=10)
    
    # Inference Layer
    inference_box = FancyBboxPatch((2, 6.5), 2, 1.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor='lightcoral', edgecolor='red')
    ax.add_patch(inference_box)
    ax.text(3, 7.25, 'Inference Layer', ha='center', va='center', fontweight='bold')
    ax.text(3, 6.8, '• Binary Classification\n• Probability Output\n• Risk Assessment', 
            ha='center', va='center', fontsize=10)
    
    # Grad-CAM Layer
    gradcam_box = FancyBboxPatch((5, 6.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightpink', edgecolor='purple')
    ax.add_patch(gradcam_box)
    ax.text(6, 7.25, 'Interpretability Layer', ha='center', va='center', fontweight='bold')
    ax.text(6, 6.8, '• Grad-CAM Visualization\n• Heatmap Generation\n• Model Explanation', 
            ha='center', va='center', fontsize=10)
    
    # UI Layer
    ui_box = FancyBboxPatch((1, 3.5), 4, 1.5, 
                           boxstyle="round,pad=0.1", 
                           facecolor='lightgray', edgecolor='black')
    ax.add_patch(ui_box)
    ax.text(3, 4.25, 'User Interface Layer', ha='center', va='center', fontweight='bold')
    ax.text(3, 3.8, '• Streamlit Web App\n• Image Upload\n• Results Visualization', 
            ha='center', va='center', fontsize=10)
    
    # Metrics Layer
    metrics_box = FancyBboxPatch((6, 3.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor='lightcyan', edgecolor='teal')
    ax.add_patch(metrics_box)
    ax.text(7, 4.25, 'Metrics Layer', ha='center', va='center', fontweight='bold')
    ax.text(7, 3.8, '• AUROC, F1-Score\n• Confusion Matrix\n• ROC Curves', 
            ha='center', va='center', fontsize=10)
    
    # Arrows showing data flow
    arrows = [
        # Data to Model
        ((1.5, 9.5), (4.5, 9.5)),
        # Model to Training
        ((4.5, 9.5), (7.5, 9.5)),
        # Training to Inference
        ((7.5, 9.5), (3, 8)),
        # Model to Inference
        ((4.5, 9.5), (3, 8)),
        # Model to Grad-CAM
        ((4.5, 9.5), (6, 8)),
        # Inference to UI
        ((3, 6.5), (3, 5)),
        # Grad-CAM to UI
        ((6, 6.5), (3, 5)),
        # Inference to Metrics
        ((3, 6.5), (7, 5))
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black")
        ax.add_patch(arrow)
    
    # Add layer labels
    ax.text(-0.5, 10.25, 'Data\nProcessing', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    ax.text(-0.5, 7.25, 'Model\nInference', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    ax.text(-0.5, 4.25, 'User\nInterface', ha='center', va='center', 
            fontsize=12, fontweight='bold', rotation=90)
    
    # Add workflow description
    ax.text(5, 1.5, 'Workflow: Data → Model Training → Inference → Grad-CAM → UI Visualization', 
            ha='center', va='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    return fig


def create_wireframe_diagram():
    """Create UI wireframe diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Medical X-ray Triage - UI Wireframe', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Main container
    main_box = FancyBboxPatch((0.5, 0.5), 11, 6.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor='white', edgecolor='black', linewidth=2)
    ax.add_patch(main_box)
    
    # Header
    header_box = FancyBboxPatch((1, 6), 10, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightblue', edgecolor='blue')
    ax.add_patch(header_box)
    ax.text(6, 6.4, '🏥 Medical X-ray Triage System', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Sidebar
    sidebar_box = FancyBboxPatch((1, 1), 3, 4.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightgray', edgecolor='gray')
    ax.add_patch(sidebar_box)
    ax.text(2.5, 5.2, 'Configuration', ha='center', va='center', fontweight='bold')
    
    # Sidebar elements
    sidebar_elements = [
        'Model Selection',
        'Image Size Slider',
        'Threshold Slider',
        'Grad-CAM Method',
        'Device Selection'
    ]
    
    for i, element in enumerate(sidebar_elements):
        y_pos = 4.5 - i * 0.6
        element_box = FancyBboxPatch((1.2, y_pos-0.2), 2.6, 0.4, 
                                    boxstyle="round,pad=0.02", 
                                    facecolor='white', edgecolor='darkgray')
        ax.add_patch(element_box)
        ax.text(2.5, y_pos, element, ha='center', va='center', fontsize=9)
    
    # Main content area
    main_content_box = FancyBboxPatch((4.5, 1), 6, 4.5, 
                                     boxstyle="round,pad=0.05", 
                                     facecolor='white', edgecolor='black')
    ax.add_patch(main_content_box)
    
    # Upload section
    upload_box = FancyBboxPatch((5, 4.5), 4.5, 0.8, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgreen', edgecolor='green')
    ax.add_patch(upload_box)
    ax.text(7.25, 4.9, '📤 Upload X-ray Image', 
            ha='center', va='center', fontweight='bold')
    
    # Image preview area
    image_box = FancyBboxPatch((5, 2.5), 4.5, 1.8, 
                              boxstyle="round,pad=0.05", 
                              facecolor='lightyellow', edgecolor='orange')
    ax.add_patch(image_box)
    ax.text(7.25, 3.4, 'Image Preview Area', 
            ha='center', va='center', fontweight='bold')
    
    # Results section
    results_box = FancyBboxPatch((5, 1.2), 4.5, 1.2, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightcoral', edgecolor='red')
    ax.add_patch(results_box)
    ax.text(7.25, 1.8, 'Analysis Results', 
            ha='center', va='center', fontweight='bold')
    
    # Results elements
    result_elements = ['Prediction', 'Confidence', 'Risk Level']
    for i, element in enumerate(result_elements):
        x_pos = 5.5 + i * 1.3
        result_element_box = FancyBboxPatch((x_pos, 1.3), 1.1, 0.6, 
                                           boxstyle="round,pad=0.02", 
                                           facecolor='white', edgecolor='darkgray')
        ax.add_patch(result_element_box)
        ax.text(x_pos + 0.55, 1.6, element, ha='center', va='center', fontsize=8)
    
    # Grad-CAM visualization area
    gradcam_box = FancyBboxPatch((10, 1), 1.3, 4.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='lightpink', edgecolor='purple')
    ax.add_patch(gradcam_box)
    ax.text(10.65, 5.5, 'Grad-CAM\nVisualization', 
            ha='center', va='center', fontweight='bold', rotation=90)
    
    # Disclaimer
    ax.text(6, 0.2, '⚠️ For Research and Educational Use Only', 
            ha='center', va='center', fontsize=10, style='italic', color='red')
    
    plt.tight_layout()
    return fig


def main():
    """Generate all documentation diagrams."""
    # Create output directory
    os.makedirs('docs', exist_ok=True)
    
    # Generate architecture diagram
    print("Generating architecture diagram...")
    arch_fig = create_architecture_diagram()
    arch_fig.savefig('docs/architecture.png', dpi=300, bbox_inches='tight')
    plt.close(arch_fig)
    print("✓ Architecture diagram saved to docs/architecture.png")
    
    # Generate wireframe diagram
    print("Generating UI wireframe...")
    wireframe_fig = create_wireframe_diagram()
    wireframe_fig.savefig('docs/wireframe.png', dpi=300, bbox_inches='tight')
    plt.close(wireframe_fig)
    print("✓ UI wireframe saved to docs/wireframe.png")
    
    print("\nAll documentation diagrams generated successfully!")


if __name__ == "__main__":
    main()


