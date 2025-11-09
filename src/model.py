"""
Model definitions for the Medical X-ray Triage project.

This module contains functions to create and load pretrained CNN models
for binary abnormality detection in chest X-rays.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_V2_S_Weights
import torch.nn.functional as F


class ChestXrayClassifier(nn.Module):
    """
    CNN classifier for chest X-ray abnormality detection.
    
    This class wraps a pretrained backbone with a custom classification head.
    """
    
    def __init__(self, backbone_name="resnet50", num_classes=1, pretrained=True, dropout_rate=0.5):
        """
        Initialize the classifier.
        
        Args:
            backbone_name (str): Name of the backbone model
            num_classes (int): Number of output classes (1 for binary classification)
            pretrained (bool): Whether to use pretrained weights
            dropout_rate (float): Dropout rate for regularization
        """
        super(ChestXrayClassifier, self).__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        
        # Load backbone
        self.backbone = self._load_backbone(backbone_name, pretrained)
        
        # Get number of features from backbone
        self.num_features = self._get_num_features()
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _load_backbone(self, backbone_name, pretrained):
        """
        Load the pretrained backbone model.
        
        Args:
            backbone_name (str): Name of the backbone model
            pretrained (bool): Whether to use pretrained weights
        
        Returns:
            torch.nn.Module: Pretrained backbone model
        """
        if backbone_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            # Remove the original classifier
            model = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
            
        elif backbone_name == "resnet50":
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            # Remove the original classifier
            model = nn.Sequential(*list(model.children())[:-2])  # Remove avgpool and fc
            
        elif backbone_name == "efficientnet_v2_s":
            model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            # Remove the original classifier
            model = nn.Sequential(*list(model.children())[:-1])  # Remove classifier
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        return model
    
    def _get_num_features(self):
        """
        Get the number of output features from the backbone.
        
        Returns:
            int: Number of features
        """
        if self.backbone_name == "resnet18":
            return 512  # ResNet18 features
        elif self.backbone_name == "resnet50":
            return 2048  # ResNet50 features
        elif self.backbone_name == "efficientnet_v2_s":
            return 1280  # EfficientNetV2-S features
        else:
            raise ValueError(f"Unknown number of features for backbone: {self.backbone_name}")
    
    def _initialize_weights(self):
        """
        Initialize the weights of the classifier head.
        """
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x):
        """
        Predict probabilities for input.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Probability predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 1:
                # Binary classification
                probs = torch.sigmoid(logits)
            else:
                # Multi-class classification
                probs = F.softmax(logits, dim=1)
        return probs


def load_backbone(model_name, pretrained=True):
    """
    Load a pretrained backbone model.
    
    Args:
        model_name (str): Name of the model to load
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: Loaded model
    """
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        # Remove the original classifier
        model = nn.Sequential(*list(model.children())[:-2])
        
    elif model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove the original classifier
        model = nn.Sequential(*list(model.children())[:-1])
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model


def create_model(model_name="resnet50", num_classes=1, pretrained=True, dropout_rate=0.5):
    """
    Create a chest X-ray classifier model.
    
    Args:
        model_name (str): Name of the backbone model
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        dropout_rate (float): Dropout rate for regularization
    
    Returns:
        ChestXrayClassifier: Created model
    """
    model = ChestXrayClassifier(
        backbone_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    
    return model


def get_model_summary(model, input_size=(3, 320, 320)):
    """
    Get a summary of the model architecture.
    
    Args:
        model (torch.nn.Module): Model to summarize
        input_size (tuple): Input size for the model
    
    Returns:
        dict: Model summary information
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, *input_size)
        output = model(dummy_input)
    
    summary = {
        'model_name': model.backbone_name if hasattr(model, 'backbone_name') else 'Unknown',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'input_size': input_size,
        'output_size': output.shape,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }
    
    return summary


def save_model(model, save_path, optimizer=None, epoch=None, metrics=None):
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): Model to save
        save_path (str): Path to save the model
        optimizer (torch.optim.Optimizer): Optimizer state (optional)
        epoch (int): Current epoch (optional)
        metrics (dict): Metrics to save (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model.backbone_name if hasattr(model, 'backbone_name') else 'unknown',
        'num_classes': model.num_classes if hasattr(model, 'num_classes') else 1,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, save_path)


def load_model(load_path, device='cpu', model_name=None):
    """
    Load model from checkpoint.
    
    Args:
        load_path (str): Path to load the model from
        device (str): Device to load the model on
        model_name (str): Model name if not in checkpoint (default: 'resnet18')
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    # Check if checkpoint is a dict with metadata or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format
        state_dict = checkpoint['model_state_dict']
        model_name = checkpoint.get('model_name', model_name or 'resnet18')
        num_classes = checkpoint.get('num_classes', 1)
        epoch = checkpoint.get('epoch', None)
        metrics = checkpoint.get('metrics', None)
    elif isinstance(checkpoint, dict) and any(k.startswith('backbone.') or k.startswith('classifier.') for k in checkpoint.keys()):
        # Direct state_dict format (saved from train.py)
        state_dict = checkpoint
        # Try to get model_name from params.yaml or metrics.json
        if model_name is None:
            import os
            import yaml
            import json
            params_path = os.path.join(os.path.dirname(load_path), 'params.yaml')
            metrics_path = os.path.join(os.path.dirname(load_path), 'metrics.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = yaml.safe_load(f)
                    model_name = params.get('model_name', 'resnet18')
            elif os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                    config = metrics_data.get('config', {})
                    model_name = config.get('model_name', 'resnet18')
            else:
                model_name = 'resnet18'
        num_classes = 1
        epoch = None
        metrics = None
    else:
        # Assume it's a state_dict
        state_dict = checkpoint
        model_name = model_name or 'resnet18'
        num_classes = 1
        epoch = None
        metrics = None
    
    # Create model
    model = create_model(model_name=model_name, num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Extract checkpoint info
    checkpoint_info = {
        'epoch': epoch,
        'metrics': metrics,
        'model_name': model_name,
        'num_classes': num_classes
    }
    
    return model, checkpoint_info


