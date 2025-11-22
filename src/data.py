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
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
                       train_split=0.70, val_split=0.15, test_split=0.15,
                       num_workers=4, use_weighted_sampling=True, random_state=42):
    """
    Create data loaders for training, validation, and testing.
    
    We use a 70/15/15 train/val/test split to ensure:
    - enough data for training,
    - a reasonably sized validation set for early stopping,
    - a held-out test set for final reporting.
    
    Args:
        labels_path (str): Path to the labels CSV file
        images_dir (str): Directory containing the images
        batch_size (int): Batch size for data loaders
        img_size (int): Size to resize images to
        train_split (float): Fraction of data for training (default: 0.70)
        val_split (float): Fraction of data for validation (default: 0.15)
        test_split (float): Fraction of data for testing (default: 0.15)
        num_workers (int): Number of workers for data loading
        use_weighted_sampling (bool): Whether to use weighted sampling for class balance
        random_state (int): Random seed for reproducibility (default: 42)
    
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
    
    print(f"Total dataset size: {len(labels)}")
    print(f"Class distribution: {dict(Counter(labels))}")
    
    # We use a 70/15/15 train/val/test split to ensure:
    # - enough data for training,
    # - a reasonably sized validation set for early stopping,
    # - a held-out test set for final reporting.
    # Split indices with stratification
    train_indices, temp_indices = train_test_split(
        indices, test_size=(val_split + test_split), 
        random_state=random_state, stratify=labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=(test_split / (val_split + test_split)),
        random_state=random_state, stratify=labels[temp_indices]
    )
    
    # Print split sizes
    print(f"\nDataset Split Sizes:")
    print(f"  Train size: {len(train_indices)}")
    print(f"  Val size: {len(val_indices)}")
    print(f"  Test size: {len(test_indices)}")
    print(f"  Total: {len(train_indices) + len(val_indices) + len(test_indices)}")
    print(f"  Train class distribution: {dict(Counter(labels[train_indices]))}")
    print(f"  Val class distribution: {dict(Counter(labels[val_indices]))}")
    print(f"  Test class distribution: {dict(Counter(labels[test_indices]))}")
    
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


class UnifiedChestXrayDataset(Dataset):
    """
    Unified dataset class that can handle images from different directories
    based on the original split (train/val/test).
    """
    def __init__(self, data_df, train_images_dir, val_images_dir, test_images_dir, transform=None):
        self.data_df = data_df.reset_index(drop=True)
        self.transform = transform
        
        # Map original splits to image directories
        self.image_dirs = {
            'train': train_images_dir,
            'val': val_images_dir,
            'test': test_images_dir
        }
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        filepath = row['filepath']
        label = row['label']
        original_split = row['original_split']
        
        # Get correct image directory based on original split
        images_dir = self.image_dirs[original_split]
        
        # Filepath already includes class folder (NORMAL/ or PNEUMONIA/)
        image_path = os.path.join(images_dir, filepath)
        
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def create_pre_split_data_loaders(
    data_dir, batch_size=8, img_size=320, num_workers=4, use_weighted_sampling=True,
    train_split=0.70, val_split=0.15, test_split=0.15, random_state=42
):
    """
    Create data loaders from pre-split NIH dataset structure with proper re-splitting.
    
    This function loads data from separate CSV files and re-splits them into proper
    train/val/test splits. Expected structure:
        data_dir/
            train_labels.csv, val_labels.csv, test_labels.csv
            train/, val/, test/ (each containing NORMAL/ and PNEUMONIA/ folders)
    
    We use a 70/15/15 train/val/test split to ensure:
    - enough data for training,
    - a reasonably sized validation set for early stopping,
    - a held-out test set for final reporting.
    
    Args:
        data_dir (str): Root directory containing the pre-split dataset
        batch_size (int): Batch size for data loaders
        img_size (int): Size to resize images to
        num_workers (int): Number of workers for data loading
        use_weighted_sampling (bool): Whether to use weighted sampling for class balance
        train_split (float): Fraction of data for training (default: 0.70)
        val_split (float): Fraction of data for validation (default: 0.15)
        test_split (float): Fraction of data for testing (default: 0.15)
        random_state (int): Random seed for reproducibility (default: 42)
    
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
    
    # Load all data and combine into a single dataset
    print("Loading and combining all data splits...")
    train_df = pd.read_csv(train_labels_path)
    val_df = pd.read_csv(val_labels_path)
    test_df = pd.read_csv(test_labels_path)
    
    # Combine all dataframes
    # Add a column to track original split for reference
    train_df['original_split'] = 'train'
    val_df['original_split'] = 'val'
    test_df['original_split'] = 'test'
    
    # Combine all data
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Get labels for stratified splitting
    all_labels = all_data['label'].values
    all_indices = np.arange(len(all_labels))
    
    print(f"Total dataset size: {len(all_data)}")
    print(f"Class distribution: {dict(Counter(all_labels))}")
    
    # We use a 70/15/15 train/val/test split to ensure:
    # - enough data for training,
    # - a reasonably sized validation set for early stopping,
    # - a held-out test set for final reporting.
    # Split indices with stratification
    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=(val_split + test_split),
        random_state=random_state,
        stratify=all_labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(test_split / (val_split + test_split)),
        random_state=random_state,
        stratify=all_labels[temp_indices]
        )
    
    # Get transforms
    train_transform = get_transforms(img_size, is_training=True)
    val_transform = get_transforms(img_size, is_training=False)
    
    # Create unified dataset
    full_dataset = UnifiedChestXrayDataset(
        all_data, 
        train_images_dir, 
        val_images_dir, 
        test_images_dir,
        transform=None
    )
    
    # Create subset datasets with transforms
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Apply transforms to subsets
    # We need to set transform on the underlying dataset
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    # Print split sizes
    print(f"\nDataset Split Sizes:")
    print(f"  Train size: {len(train_dataset)}")
    print(f"  Val size: {len(val_dataset)}")
    print(f"  Test size: {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    # Calculate class weights for training set
    train_labels = all_labels[train_indices]
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)
    class_weights = {
        class_id: total_samples / (len(class_counts) * count)
        for class_id, count in class_counts.items()
    }
    
    print(f"  Train class distribution: {dict(Counter(train_labels))}")
    print(f"  Val class distribution: {dict(Counter(all_labels[val_indices]))}")
    print(f"  Test class distribution: {dict(Counter(all_labels[test_indices]))}")
    
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


