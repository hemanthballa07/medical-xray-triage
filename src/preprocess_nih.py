"""
Preprocessing script for NIH Chest X-ray dataset.

This script:
1. Reads Data_Entry_2017.csv
2. Filters for "Pneumonia" and "No Finding" labels
3. Splits data into train/val/test (70/15/15)
4. Resizes images to 320x320
5. Organizes into class folders (NORMAL and PNEUMONIA)
6. Generates labels.csv files for each split
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse
from typing import Tuple, Dict


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess NIH Chest X-ray dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/nih_chest_xray",
        help="Directory containing NIH dataset (with Data_Entry_2017.csv and images)"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing original images (default: data_dir/images)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for processed data (default: data_dir)"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=320,
        help="Target image size (square)"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.70,
        help="Training set fraction"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Validation set fraction"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.15,
        help="Test set fraction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_resize",
        action="store_true",
        help="Skip image resizing (assumes images are already 320x320)"
    )
    return parser.parse_args()


def load_and_filter_data(csv_path: str) -> pd.DataFrame:
    """
    Load Data_Entry_2017.csv and filter for Pneumonia and No Finding.
    
    Args:
        csv_path: Path to Data_Entry_2017.csv
    
    Returns:
        Filtered DataFrame with columns: Image Index, Finding Labels
    """
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Finding Labels'].isin(['No Finding', 'Pneumonia'])].copy()
    print(f"Filtered to {len(df_filtered)} images (No Finding or Pneumonia)")
    
    return df_filtered


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create labels DataFrame with filename and label columns.
    
    Args:
        df: DataFrame with Image Index and Finding Labels
    
    Returns:
        DataFrame with columns: filename, label (0=Normal, 1=Pneumonia)
    """
    labels_df = pd.DataFrame({
        'filename': df['Image Index'].values,
        'label': (df['Finding Labels'] == 'Pneumonia').astype(int)
    })
    
    return labels_df


def find_image_files(images_dir: str, filenames: list) -> Dict[str, str]:
    """
    Find all image files matching the filenames in the dataset.
    
    Args:
        images_dir: Directory containing images (may have subdirectories)
        filenames: List of image filenames to find
    
    Returns:
        Dictionary mapping filename to full path
    """
    print(f"Searching for images in: {images_dir}")
    
    # Find all image files (check common extensions)
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = {}
    
    # Walk through directory to find images
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if any(file.endswith(ext) for ext in image_extensions):
                # Check if this file is in our list
                if file in filenames:
                    image_files[file] = os.path.join(root, file)
    
    print(f"Found {len(image_files)} out of {len(filenames)} images")
    
    if len(image_files) < len(filenames):
        missing = set(filenames) - set(image_files.keys())
        print(f"Warning: {len(missing)} images not found")
    
    return image_files


def resize_and_save_image(
    input_path: str,
    output_path: str,
    img_size: int = 320
) -> bool:
    """
    Resize image to target size and save.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        img_size: Target image size (square)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Open and resize image
        img = Image.open(input_path).convert('RGB')
        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        
        # Save resized image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_resized.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"Error processing {os.path.basename(input_path)}: {e}")
        return False


def process_dataset(
    labels_df: pd.DataFrame,
    image_files: Dict[str, str],
    output_base_dir: str,
    split_name: str,
    img_size: int = 320,
    skip_resize: bool = False
) -> pd.DataFrame:
    """
    Process images for a specific split (train/val/test).
    
    Args:
        labels_df: DataFrame with filename and label columns
        image_files: Dictionary mapping filename to input path
        output_base_dir: Base output directory
        split_name: Name of split (train/val/test)
        img_size: Target image size
        skip_resize: Whether to skip resizing
    
    Returns:
        Updated labels_df with filepath column
    """
    print(f"\nProcessing {split_name} split...")
    
    # Create output directories
    normal_dir = os.path.join(output_base_dir, split_name, 'NORMAL')
    pneumonia_dir = os.path.join(output_base_dir, split_name, 'PNEUMONIA')
    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(pneumonia_dir, exist_ok=True)
    
    # Process each image
    processed_labels = []
    missing_count = 0
    
    for idx, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Processing {split_name}"):
        filename = row['filename']
        label = row['label']
        
        # Find source image
        if filename not in image_files:
            missing_count += 1
            continue
        
        source_path = image_files[filename]
        
        # Determine output directory and path
        class_dir = pneumonia_dir if label == 1 else normal_dir
        output_path = os.path.join(class_dir, filename)
        
        # Resize and save image
        if not skip_resize:
            success = resize_and_save_image(source_path, output_path, img_size)
            if not success:
                missing_count += 1
                continue
        else:
            # Just copy if skip_resize (or create symlink)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if not os.path.exists(output_path):
                import shutil
                shutil.copy2(source_path, output_path)
        
        # Create filepath relative to the split's images directory
        # Format: CLASS/filename (since images_dir will point to the split folder)
        filepath = os.path.join('PNEUMONIA' if label == 1 else 'NORMAL', filename)
        
        processed_labels.append({
            'filepath': filepath,
            'label': label
        })
    
    if missing_count > 0:
        print(f"Warning: {missing_count} images could not be processed")
    
    processed_df = pd.DataFrame(processed_labels)
    print(f"Processed {len(processed_df)} images for {split_name}: {(processed_df['label'] == 0).sum()} Normal, {(processed_df['label'] == 1).sum()} Pneumonia")
    
    return processed_df


def main():
    """Main preprocessing pipeline."""
    args = parse_arguments()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Set up paths
    data_dir = Path(args.data_dir).resolve()
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / 'images'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    
    # Check if CSV exists
    csv_path = data_dir / 'Data_Entry_2017.csv'
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data_Entry_2017.csv not found in {data_dir}\n"
            f"Please download the dataset from Kaggle and extract it to {data_dir}"
        )
    
    # Check if images directory exists
    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images directory not found: {images_dir}\n"
            f"Please extract image zip files (images_001.zip to images_012.zip) to {images_dir}"
        )
    
    # Load and filter data
    df_filtered = load_and_filter_data(str(csv_path))
    
    # Find all image files first
    all_filenames = df_filtered['Image Index'].unique().tolist()
    image_files = find_image_files(str(images_dir), all_filenames)
    
    # Create labels only for images that were found
    df_found = df_filtered[df_filtered['Image Index'].isin(image_files.keys())]
    labels_df = create_labels(df_found)
    
    print(f"Final dataset: {len(labels_df)} images ({(labels_df['label'] == 0).sum()} Normal, {(labels_df['label'] == 1).sum()} Pneumonia)")
    print(f"Splitting: {args.train_split:.0%} train, {args.val_split:.0%} val, {args.test_split:.0%} test")
    
    # First split: train vs temp (val + test)
    train_df, temp_df = train_test_split(
        labels_df,
        test_size=(args.val_split + args.test_split),
        random_state=args.seed,
        stratify=labels_df['label']
    )
    
    # Second split: val vs test
    val_size = args.val_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        random_state=args.seed,
        stratify=temp_df['label']
    )
    
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Process each split
    train_processed = process_dataset(
        train_df, image_files, str(output_dir), 'train',
        img_size=args.img_size, skip_resize=args.skip_resize
    )
    val_processed = process_dataset(
        val_df, image_files, str(output_dir), 'val',
        img_size=args.img_size, skip_resize=args.skip_resize
    )
    test_processed = process_dataset(
        test_df, image_files, str(output_dir), 'test',
        img_size=args.img_size, skip_resize=args.skip_resize
    )
    
    # Save labels.csv files
    train_labels_path = output_dir / 'train_labels.csv'
    val_labels_path = output_dir / 'val_labels.csv'
    test_labels_path = output_dir / 'test_labels.csv'
    all_labels_path = output_dir / 'labels.csv'
    
    train_processed.to_csv(train_labels_path, index=False)
    val_processed.to_csv(val_labels_path, index=False)
    test_processed.to_csv(test_labels_path, index=False)
    
    # Create unified labels.csv (for compatibility with existing code)
    # This will use the train split by default
    train_processed.to_csv(all_labels_path, index=False)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Output: {output_dir}/train, {output_dir}/val, {output_dir}/test")
    print(f"Labels: {train_labels_path}, {val_labels_path}, {test_labels_path}")
    print(f"\nUsage: python -m src.train --data_dir {output_dir}")


if __name__ == "__main__":
    main()

