"""
Data loading and preprocessing for the Medical X-ray Triage project.

This module contains the ChestXrayDataset class and data transformation utilities.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


class ChestXrayDataset(Dataset):
    """
    Custom dataset for chest X-ray images.
    
    Args:
        labels_path (str): Path to the labels CSV file
        images_dir (str): Directory containing the images
        transform (callable): Optional transform to be applied on a sample
    """
    
    def __init__(self, labels_path, images_dir, transform=None):
        self.labels_df = pd.read_csv(labels_path)
        self.images_dir = images_dir
        self.transform = transform
        
        # Validate that all images exist
        self._validate_images()
    
    def _validate_images(self):
        """Validate that all images referenced in labels exist."""
        missing_images = []
        for idx, row in self.labels_df.iterrows():
            # Handle both cases: with and without 'images/' prefix
            filepath = row['filepath']
            if filepath.startswith('images/'):
                # Remove 'images/' prefix if present
                filepath = filepath[7:]  # Remove 'images/' (7 characters)
            
            image_path = os.path.join(self.images_dir, filepath)
            if not os.path.exists(image_path):
                missing_images.append(image_path)
        
        if missing_images:
            raise FileNotFoundError(f"Missing images: {missing_images}")
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            tuple: (image_tensor, label)
        """
        row = self.labels_df.iloc[idx]
        
        # Load image
        filepath = row['filepath']
        if filepath.startswith('images/'):
            # Remove 'images/' prefix if present
            filepath = filepath[7:]  # Remove 'images/' (7 characters)
        
        image_path = os.path.join(self.images_dir, filepath)
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        return image, label
    
    def get_class_counts(self):
        """
        Get the count of samples for each class.
        
        Returns:
            dict: Dictionary with class counts
        """
        return dict(Counter(self.labels_df['label']))


def get_transforms(img_size=320, is_training=True):
    """
    Get data transforms for training or validation.
    
    Args:
        img_size (int): Size to resize images to
        is_training (bool): Whether to apply training augmentations
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(labels_path, images_dir, batch_size=8, img_size=320, 
                       train_split=0.6, val_split=0.2, test_split=0.2,
                       num_workers=4, use_weighted_sampling=True):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        labels_path (str): Path to the labels CSV file
        images_dir (str): Directory containing the images
        batch_size (int): Batch size for data loaders
        img_size (int): Size to resize images to
        train_split (float): Fraction of data for training
        val_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
        num_workers (int): Number of workers for data loading
        use_weighted_sampling (bool): Whether to use weighted sampling for class balance
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    # Load the full dataset
    full_dataset = ChestXrayDataset(
        labels_path=labels_path,
        images_dir=images_dir,
        transform=None  # We'll apply transforms in data loaders
    )
    
    # Get labels for splitting
    labels = full_dataset.labels_df['label'].values
    indices = np.arange(len(labels))
    
    # Split indices
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_split + test_split), 
        random_state=42, stratify=labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(test_split / (val_split + test_split)),
        random_state=42, stratify=labels[temp_indices]
    )
    
    # Create datasets with transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Calculate class weights for imbalanced datasets
    class_counts = Counter(labels[train_indices])
    total_samples = len(train_indices)
    class_weights = {
        class_id: total_samples / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    
    # Create weighted sampler for training
    train_sampler = None
    if use_weighted_sampling and len(class_counts) > 1:
        sample_weights = [class_weights[labels[idx]] for idx in train_indices]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, class_weights


def get_simple_data_loader(labels_path, images_dir, batch_size=8, img_size=320,
                          is_training=True, num_workers=4):
    """
    Create a simple data loader without train/val/test splits.
    Useful for small datasets or when you want to use all data.
    
    Args:
        labels_path (str): Path to the labels CSV file
        images_dir (str): Directory containing the images
        batch_size (int): Batch size for data loader
        img_size (int): Size to resize images to
        is_training (bool): Whether to apply training transforms
        num_workers (int): Number of workers for data loading
    
    Returns:
        torch.utils.data.DataLoader: Data loader
    """
    transform = get_transforms(img_size, is_training=is_training)
    
    dataset = ChestXrayDataset(
        labels_path=labels_path,
        images_dir=images_dir,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return data_loader, dataset


def print_dataset_info(labels_path, images_dir):
    """
    Print information about the dataset.
    
    Args:
        labels_path (str): Path to the labels CSV file
        images_dir (str): Directory containing the images
    """
    dataset = ChestXrayDataset(labels_path, images_dir)
    
    print("Dataset Information")
    print("=" * 50)
    print(f"Total samples: {len(dataset)}")
    
    class_counts = dataset.get_class_counts()
    print(f"Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        class_name = "Normal" if class_id == 0 else "Abnormal"
        percentage = (count / len(dataset)) * 100
        print(f"  {class_name} (class {class_id}): {count} ({percentage:.1f}%)")
    
    # Check image properties
    sample_image, _ = dataset[0]
    if isinstance(sample_image, torch.Tensor):
        print(f"Image shape: {sample_image.shape}")
        print(f"Image dtype: {sample_image.dtype}")
        print(f"Image range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")


if __name__ == "__main__":
    # Test data loading
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.append(str(Path(__file__).parent))
    
    from config import get_config
    
    # Get configuration
    config = get_config()
    
    # Test with sample data
    try:
        print_dataset_info(config['labels_path'], config['images_dir'])
        
        # Test data loaders
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            labels_path=config['labels_path'],
            images_dir=config['images_dir'],
            batch_size=config['batch_size'],
            img_size=config['img_size'],
            num_workers=config['num_workers']
        )
        
        print(f"\nData Loader Information")
        print("=" * 50)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Class weights: {class_weights}")
        
        # Test a batch
        for images, labels in train_loader:
            print(f"\nBatch Information")
            print("=" * 30)
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels: {labels}")
            break
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python src/make_sample_data.py' first to create sample data.")
