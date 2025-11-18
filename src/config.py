"""
Configuration management for Medical X-ray Triage project.

This module handles command-line arguments and default configuration parameters.
"""

import argparse
import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_config():
    """Get configuration from command line arguments and defaults."""
    parser = argparse.ArgumentParser(
        description="Medical X-ray Triage with CNNs, Grad-CAM, and Streamlit UI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file"
    )
    
    # Data configuration
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="./data/sample",
        help="Directory containing training data"
    )
    parser.add_argument(
        "--images_dir", 
        type=str, 
        default=None,
        help="Directory containing images (default: data_dir/images or data_dir/train for NIH dataset)"
    )
    parser.add_argument(
        "--labels_path", 
        type=str, 
        default=None,
        help="Path to labels CSV file (default: data_dir/labels.csv or data_dir/train_labels.csv for NIH dataset)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./results",
        help="Directory to save model outputs and results"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=320,
        help="Image size for training (square images)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="resnet18",
        choices=["resnet18", "resnet50", "efficientnet_v2_s"],
        help="Pretrained model backbone"
    )
    parser.add_argument(
        "--pretrained", 
        action="store_true", 
        default=True,
        help="Use pretrained weights"
    )
    
    # Training configuration
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=25,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4,
        help="Weight decay for regularization"
    )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=8,
        help="Early stopping patience"
    )
    
    # Evaluation configuration
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./results/best.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.5,
        help="Classification threshold"
    )
    
    # System configuration
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1337,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    # Logging and saving
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=1,
        help="Frequency to save model checkpoints"
    )
    parser.add_argument(
        "--log_freq", 
        type=int, 
        default=10,
        help="Frequency to log training metrics"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=False,
        help="Enable verbose logging"
    )
    
    # Grad-CAM configuration
    parser.add_argument(
        "--cam_layer", 
        type=str, 
        default="auto",
        help="Layer name for Grad-CAM (auto to use default for model)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image for Grad-CAM visualization"
    )
    parser.add_argument(
        "--cam_method", 
        type=str, 
        default="GradCAM",
        choices=["GradCAM", "GradCAM++", "XGradCAM"],
        help="Grad-CAM method to use"
    )
    
    args = parser.parse_args()
    
    # Load YAML config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Override args with YAML config
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Post-process arguments
    config = post_process_config(args)
    
    return config


def post_process_config(args) -> Dict[str, Any]:
    """Post-process configuration arguments."""
    # Convert to dictionary for easier access
    config = vars(args)
    
    # Ensure directories exist
    os.makedirs(config["data_dir"], exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Set device
    if config["device"] == "auto":
        import torch
        if torch.cuda.is_available():
            config["device"] = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config["device"] = "mps"
        else:
            config["device"] = "cpu"
    
    # Set default CAM layer based on model
    if config["cam_layer"] == "auto":
        if config["model_name"] == "resnet50":
            config["cam_layer"] = "layer4"
        elif config["model_name"] == "efficientnet_v2_s":
            config["cam_layer"] = "features.7"
        else:
            config["cam_layer"] = "features"
    
    # Add computed paths
    # Handle NIH dataset structure (has train/val/test folders)
    data_dir = config["data_dir"]
    train_labels_path = os.path.join(data_dir, "train_labels.csv")
    train_images_dir = os.path.join(data_dir, "train")
    
    # Check if NIH dataset structure exists
    is_nih_dataset = (
        os.path.exists(train_labels_path) and 
        os.path.exists(train_images_dir)
    )
    
    # Set labels_path
    if config["labels_path"] is None:
        if is_nih_dataset:
            # Default to train split for NIH dataset
            config["labels_path"] = train_labels_path
        else:
            config["labels_path"] = os.path.join(data_dir, "labels.csv")
    else:
        config["labels_path"] = config["labels_path"]
    
    # Set images_dir
    if config["images_dir"] is None:
        if is_nih_dataset:
            # Default to train split for NIH dataset
            config["images_dir"] = train_images_dir
        else:
            config["images_dir"] = os.path.join(data_dir, "images")
    else:
        config["images_dir"] = config["images_dir"]
    
    config["best_model_path"] = os.path.join(config["output_dir"], "best.pt")
    config["metrics_path"] = os.path.join(config["output_dir"], "metrics.json")
    
    return config


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration in a readable format."""
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    
    for key, value in config.items():
        print(f"{key:20}: {value}")
    
    print("=" * 50)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Configuration saved to: {save_path}")


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print_config(config)
