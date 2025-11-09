"""
Model interpretation using Grad-CAM for the Medical X-ray Triage project.

This module provides functionality to generate Grad-CAM visualizations
to understand what regions of the X-ray images the model focuses on.
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config import get_config
from model import create_model
from data import get_transforms
from utils import seed_everything, get_device

try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    print("Warning: pytorch-grad-cam not installed. Install with: pip install pytorch-grad-cam")
    GradCAM = None


class ChestXrayGradCAMWrapper:
    """
    Wrapper class to make the model compatible with pytorch-grad-cam.
    """
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def __call__(self, input_tensor):
        """
        Forward pass through the model.
        
        Args:
            input_tensor: Input tensor
        
        Returns:
            torch.Tensor: Model output
        """
        return self.model(input_tensor)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
        return self
    
    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()
    
    def named_parameters(self):
        """Return named model parameters."""
        return self.model.named_parameters()
    
    def modules(self):
        """Return model modules."""
        return self.model.modules()
    
    def named_modules(self):
        """Return named model modules."""
        return self.model.named_modules()


def get_target_layers(model, model_name):
    """
    Get the target layers for Grad-CAM based on the model architecture.
    
    Args:
        model: PyTorch model
        model_name: Name of the model
    
    Returns:
        list: List of target layers
    """
    if model_name == "resnet18":
        # For ResNet18, target the last convolutional layer
        target_layers = [model.backbone[-1]]
    elif model_name == "resnet50":
        # For ResNet50, target the last convolutional layer
        target_layers = [model.backbone[-1]]
    elif model_name == "efficientnet_v2_s":
        # For EfficientNetV2-S, target the last feature layer
        target_layers = [model.backbone[-1]]
    else:
        # Default to the last layer of the backbone
        target_layers = [model.backbone[-1]]
    
    return target_layers


def generate_gradcam(image_path, model, model_name, device, cam_method="GradCAM", 
                    save_path=None, target_class=None):
    """
    Generate Grad-CAM visualization for a single image.
    
    Args:
        image_path: Path to the input image
        model: Trained PyTorch model
        model_name: Name of the model
        device: Device to run inference on
        cam_method: Grad-CAM method to use
        save_path: Path to save the visualization
        target_class: Target class for Grad-CAM (None for automatic)
    
    Returns:
        tuple: (original_image, gradcam_image, prediction_prob)
    """
    if GradCAM is None:
        raise ImportError("pytorch-grad-cam is required for Grad-CAM visualization")
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = get_transforms(img_size=320, is_training=False)
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prediction_prob = torch.sigmoid(logits).item()
    
    # Determine target class if not specified
    if target_class is None:
        target_class = 1 if prediction_prob > 0.5 else 0
    
    # Create Grad-CAM wrapper
    wrapped_model = ChestXrayGradCAMWrapper(model)
    
    # Get target layers
    target_layers = get_target_layers(model, model_name)
    
    # Build model graph
    with torch.no_grad():
        _ = model(torch.randn(1, 3, 320, 320).to(device))
    
    # Create Grad-CAM object
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    
    # Generate Grad-CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    
    # Convert original image to numpy array for overlay (range 0-1)
    img_float01 = np.array(original_image.resize((320, 320))) / 255.0
    
    # Create Grad-CAM visualization
    overlay = show_cam_on_image(img_float01, grayscale_cam, use_rgb=True)
    
    # Save if path provided
    if save_path:
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image")
        plt.axis('off')
        
        # Grad-CAM heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(grayscale_cam, cmap='jet')
        plt.title(f"Grad-CAM Heatmap")
        plt.colorbar()
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title(f"Grad-CAM Overlay\nPrediction: {prediction_prob:.3f}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Grad-CAM visualization saved to: {save_path}")
    
    return original_image, gradcam_image, prediction_prob


def batch_gradcam(image_paths, model, model_name, device, cam_method="GradCAM", 
                 output_dir="./results", prefix="cam"):
    """
    Generate Grad-CAM visualizations for multiple images.
    
    Args:
        image_paths: List of image paths
        model: Trained PyTorch model
        model_name: Name of the model
        device: Device to run inference on
        cam_method: Grad-CAM method to use
        output_dir: Directory to save visualizations
        prefix: Prefix for saved files
    
    Returns:
        list: List of results for each image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        # Generate save path
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(output_dir, f"{prefix}_{image_name}.png")
        
        try:
            original_image, gradcam_image, prediction_prob = generate_gradcam(
                image_path=image_path,
                model=model,
                model_name=model_name,
                device=device,
                cam_method=cam_method,
                save_path=save_path
            )
            
            results.append({
                'image_path': image_path,
                'save_path': save_path,
                'prediction_prob': prediction_prob,
                'prediction_class': 1 if prediction_prob > 0.5 else 0,
                'success': True
            })
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e),
                'success': False
            })
    
    return results


def compare_cam_methods(image_path, model, model_name, device, output_dir="./results"):
    """
    Compare different Grad-CAM methods on a single image.
    
    Args:
        image_path: Path to the input image
        model: Trained PyTorch model
        model_name: Name of the model
        device: Device to run inference on
        output_dir: Directory to save visualizations
    
    Returns:
        dict: Results for each CAM method
    """
    if GradCAM is None:
        raise ImportError("pytorch-grad-cam is required for Grad-CAM visualization")
    
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ["GradCAM", "GradCAMPlusPlus", "XGradCAM"]
    results = {}
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert('RGB')
    transform = get_transforms(img_size=320, is_training=False)
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prediction_prob = torch.sigmoid(logits).item()
    
    # Create figure for comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')
    
    # Generate Grad-CAM for each method
    for i, method in enumerate(methods):
        try:
            _, gradcam_image, _ = generate_gradcam(
                image_path=image_path,
                model=model,
                model_name=model_name,
                device=device,
                cam_method=method,
                save_path=None
            )
            
            # Display heatmap
            wrapped_model = ChestXrayGradCAMWrapper(model)
            target_layers = get_target_layers(model, model_name)
            
            if method == "GradCAM":
                cam = GradCAM(model=wrapped_model, target_layers=target_layers)
            elif method == "GradCAMPlusPlus":
                cam = GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)
            elif method == "XGradCAM":
                cam = XGradCAM(model=wrapped_model, target_layers=target_layers)
            
            targets = [ClassifierOutputTarget(1 if prediction_prob > 0.5 else 0)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            axes[0, i+1].imshow(grayscale_cam, cmap='jet')
            axes[0, i+1].set_title(f"{method} Heatmap")
            axes[0, i+1].axis('off')
            
            # Display overlay
            axes[1, i+1].imshow(gradcam_image)
            axes[1, i+1].set_title(f"{method} Overlay")
            axes[1, i+1].axis('off')
            
            results[method] = {
                'gradcam_image': gradcam_image,
                'success': True
            }
            
        except Exception as e:
            axes[0, i+1].text(0.5, 0.5, f"Error:\n{str(e)}", 
                            ha='center', va='center', transform=axes[0, i+1].transAxes)
            axes[0, i+1].set_title(f"{method} (Error)")
            axes[1, i+1].axis('off')
            
            results[method] = {
                'error': str(e),
                'success': False
            }
    
    # Add prediction info
    fig.suptitle(f"Grad-CAM Comparison\nPrediction: {prediction_prob:.3f} ({'Abnormal' if prediction_prob > 0.5 else 'Normal'})", 
                fontsize=16)
    
    plt.tight_layout()
    
    # Save comparison
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"cam_comparison_{image_name}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Grad-CAM comparison saved to: {save_path}")
    
    return results


def main():
    """Main function for Grad-CAM interpretation."""
    # Get configuration
    config = get_config()
    
    print("Medical X-ray Triage - Grad-CAM Interpretation")
    print("=" * 50)
    print(f"Model path: {config['model_path']}")
    print(f"Device: {config['device']}")
    print(f"CAM method: {config['cam_method']}")
    print()
    
    # Set seed for reproducibility
    seed_everything(config['seed'])
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model file not found at {config['model_path']}")
        print("Please train a model first using: python src/train.py")
        return
    
    # Check if sample data exists
    if not os.path.exists(config['labels_path']):
        print("Sample data not found. Generating sample data...")
        from make_sample_data import create_sample_dataset
        create_sample_dataset(config['data_dir'])
        print()
    
    # Get device
    if config.get('device', 'auto') == 'auto':
        device = get_device()
    else:
        device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model = create_model(config['model_name'], num_classes=1, pretrained=False)
    model.load_state_dict(torch.load(config['model_path'], map_location="cpu"))
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Model: {config['model_name']}")
    
    # Get list of images to process
    if hasattr(config, 'image') and config['image']:
        # Single image specified
        image_paths = [config['image']]
    else:
        # Process all images in directory
        images_dir = config['images_dir']
        image_extensions = ['.png', '.jpg', '.jpeg']
        
        image_paths = []
        for ext in image_extensions:
            image_paths.extend([os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                               if f.lower().endswith(ext)])
    
    if not image_paths:
        print(f"No images found in {config.get('image', config['images_dir'])}")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Generate Grad-CAM for all images
    results = batch_gradcam(
        image_paths=image_paths,
        model=model,
        model_name=config['model_name'],
        device=device,
        cam_method=config['cam_method'],
        output_dir=config['output_dir']
    )
    
    # Print results summary
    print("\nGrad-CAM Generation Summary:")
    print("=" * 40)
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nPredictions:")
        for result in successful:
            image_name = os.path.basename(result['image_path'])
            prob = result['prediction_prob']
            pred_class = "Abnormal" if result['prediction_class'] == 1 else "Normal"
            print(f"  {image_name}: {prob:.3f} ({pred_class})")
    
    if failed:
        print(f"\nFailed images:")
        for result in failed:
            image_name = os.path.basename(result['image_path'])
            print(f"  {image_name}: {result['error']}")
    
    print(f"\nGrad-CAM visualizations saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
