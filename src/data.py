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


def create_pre_split_data_loaders(
    data_dir, batch_size=8, img_size=320, num_workers=4, use_weighted_sampling=True
):
    """
    Create data loaders from pre-split NIH dataset structure.
    
    This function loads train, val, and test splits from separate CSV files
    and image directories. Expected structure:
        data_dir/
            train_labels.csv, val_labels.csv, test_labels.csv
            train/, val/, test/ (each containing NORMAL/ and PNEUMONIA/ folders)
    
    Args:
        data_dir (str): Root directory containing the pre-split dataset
        batch_size (int): Batch size for data loaders
        img_size (int): Size to resize images to
        num_workers (int): Number of workers for data loading
        use_weighted_sampling (bool): Whether to use weighted sampling for class balance
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    import os
    
    # Define paths
    train_labels_path = os.path.join(data_dir, "train_labels.csv")
    val_labels_path = os.path.join(data_dir, "val_labels.csv")
    test_labels_path = os.path.join(data_dir, "test_labels.csv")
    
    train_images_dir = os.path.join(data_dir, "train")
    val_images_dir = os.path.join(data_dir, "val")
    test_images_dir = os.path.join(data_dir, "test")
    
    # Check if all files exist
    if not all(os.path.exists(p) for p in [train_labels_path, val_labels_path, test_labels_path]):
        raise FileNotFoundError(
            f"Pre-split dataset not found in {data_dir}. "
            f"Expected files: train_labels.csv, val_labels.csv, test_labels.csv"
        )
    
    # Get transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_labels_path, train_images_dir, transform=train_transform)
    val_dataset = ChestXrayDataset(val_labels_path, val_images_dir, transform=val_transform)
    test_dataset = ChestXrayDataset(test_labels_path, test_images_dir, transform=val_transform)
    
    # Calculate class weights for training set
    train_labels = train_dataset.labels_df['label'].values
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = {
        class_id: total_samples / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    
    # Create weighted sampler for training
    train_sampler = None
    if use_weighted_sampling and len(class_counts) > 1:
        sample_weights = [class_weights[label] for label in train_labels]
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


