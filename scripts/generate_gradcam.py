#!/usr/bin/env python3
"""Generate Grad-CAM visualization for an abnormal case."""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import get_config
from model import create_model
from data import get_transforms
from utils import get_device

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Error: pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")
    sys.exit(1)


def generate_gradcam_for_abnormal(image_path, model_path, output_path, model_name="resnet18", device=None):
    """Generate Grad-CAM visualization for an abnormal case."""
    
    # Get device
    if device is None:
        device = get_device()
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = create_model(model_name, num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load and preprocess image
    print(f"Loading image: {image_path}")
    original_image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_transforms(img_size=320, is_training=False)
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get model prediction
    with torch.no_grad():
        logits = model(input_tensor)
        prediction_prob = torch.sigmoid(logits)[0, 0].item()
    
    print(f"Prediction probability: {prediction_prob:.4f} ({'Abnormal' if prediction_prob > 0.5 else 'Normal'})")
    
    # Get target layers for ResNet18
    if model_name.lower().startswith("resnet"):
        # For ResNet, target the last convolutional layer
        target_layers = [model.backbone[-1][-1].conv2]
    else:
        target_layers = [model.backbone[-1]]
    
    print(f"Target layers: {[str(l) for l in target_layers]}")
    
    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Set target for abnormal class (index 0 for single-logit binary model)
    target_idx = 0
    targets = [ClassifierOutputTarget(target_idx)]
    
    # Generate Grad-CAM
    print("Generating Grad-CAM visualization...")
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # Get first (and only) result
    
    # Convert original image to numpy array for overlay (range 0-1)
    img_resized = original_image.resize((320, 320))
    img_float01 = np.array(img_resized) / 255.0
    
    # Create Grad-CAM visualization overlay
    overlay = show_cam_on_image(img_float01, grayscale_cam, use_rgb=True)
    
    # Create figure with three panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    im = axes[1].imshow(grayscale_cam, cmap='jet')
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title(f"Grad-CAM Overlay\nPrediction: {prediction_prob:.3f} ({'Abnormal' if prediction_prob > 0.5 else 'Normal'})", 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM visualization saved to: {output_path}")


if __name__ == "__main__":
    # Find an abnormal case image
    project_root = Path(__file__).parent.parent
    abnormal_dir = project_root / "data/chest_xray/test/PNEUMONIA"
    
    if not abnormal_dir.exists():
        print(f"Error: Abnormal images directory not found: {abnormal_dir}")
        sys.exit(1)
    
    # Find first abnormal image
    abnormal_images = [f for f in os.listdir(abnormal_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not abnormal_images:
        print(f"Error: No abnormal images found in {abnormal_dir}")
        sys.exit(1)
    
    image_path = abnormal_dir / abnormal_images[0]
    model_path = project_root / "results/best.pt"
    output_path = project_root / "results/sample_gradcam.png"
    
    print("=" * 60)
    print("Generating Grad-CAM for Abnormal Case")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print()
    
    generate_gradcam_for_abnormal(
        image_path=str(image_path),
        model_path=str(model_path),
        output_path=str(output_path),
        model_name="resnet18"
    )
    
    print("\nDone!")

