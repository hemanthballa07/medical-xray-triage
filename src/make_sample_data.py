"""
Generate synthetic sample data for the Medical X-ray Triage project.

This script creates 4 synthetic chest X-ray images and corresponding labels
for testing and demonstration purposes.
"""

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import argparse
from pathlib import Path


def create_synthetic_xray(width=320, height=320, is_abnormal=False, seed=42):
    """
    Create a synthetic X-ray image.
    
    Args:
        width (int): Image width
        height (int): Image height
        is_abnormal (bool): Whether to create an abnormal X-ray
        seed (int): Random seed for reproducibility
    
    Returns:
        PIL.Image: Synthetic X-ray image
    """
    np.random.seed(seed)
    
    # Create base gradient background (simulating chest cavity)
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Create chest cavity gradient
    for y in range(height):
        # Create a gradient from dark edges to lighter center
        center_distance = abs(y - height // 2)
        max_distance = height // 2
        
        # Base intensity (darker at edges, lighter in center)
        base_intensity = int(80 + (max_distance - center_distance) * 50 / max_distance)
        
        for x in range(width):
            # Add horizontal gradient variation
            x_distance = abs(x - width // 2)
            x_max_distance = width // 2
            x_variation = int((x_max_distance - x_distance) * 30 / x_max_distance)
            
            # Add some noise
            noise = np.random.randint(-10, 11)
            
            intensity = max(0, min(255, base_intensity + x_variation + noise))
            img.putpixel((x, y), intensity)
    
    # Add rib-like structures (horizontal lines)
    for i in range(3, 8):
        y = height * i // 10
        for x in range(width):
            current_intensity = img.getpixel((x, y))
            # Make ribs slightly brighter
            new_intensity = min(255, current_intensity + 20)
            img.putpixel((x, y), new_intensity)
    
    # Add lung fields (darker regions)
    # Left lung
    left_lung_center = (width // 4, height // 2)
    for y in range(height // 4, 3 * height // 4):
        for x in range(width // 8, 3 * width // 8):
            distance = ((x - left_lung_center[0]) ** 2 + (y - left_lung_center[1]) ** 2) ** 0.5
            if distance < width // 6:
                current_intensity = img.getpixel((x, y))
                new_intensity = max(0, current_intensity - 30)
                img.putpixel((x, y), new_intensity)
    
    # Right lung
    right_lung_center = (3 * width // 4, height // 2)
    for y in range(height // 4, 3 * height // 4):
        for x in range(5 * width // 8, 7 * width // 8):
            distance = ((x - right_lung_center[0]) ** 2 + (y - right_lung_center[1]) ** 2) ** 0.5
            if distance < width // 6:
                current_intensity = img.getpixel((x, y))
                new_intensity = max(0, current_intensity - 30)
                img.putpixel((x, y), new_intensity)
    
    # Add abnormality if requested
    if is_abnormal:
        # Add a circular opacity (simulating pneumonia/consolidation)
        abnormality_center = (width // 2 + np.random.randint(-width // 8, width // 8),
                             height // 2 + np.random.randint(-height // 8, height // 8))
        abnormality_radius = width // 8
        
        for y in range(height):
            for x in range(width):
                distance = ((x - abnormality_center[0]) ** 2 + (y - abnormality_center[1]) ** 2) ** 0.5
                if distance < abnormality_radius:
                    current_intensity = img.getpixel((x, y))
                    # Make abnormality brighter (more opaque)
                    new_intensity = min(255, current_intensity + 40)
                    img.putpixel((x, y), new_intensity)
    
    # Add some smoothing to make it look more realistic
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img


def create_sample_dataset(output_dir="./data/sample", num_normal=2, num_abnormal=2):
    """
    Create the complete sample dataset.
    
    Args:
        output_dir (str): Output directory for the dataset
        num_normal (int): Number of normal images to create
        num_abnormal (int): Number of abnormal images to create
    """
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Create labels dataframe
    labels_data = []
    
    # Create normal images
    for i in range(num_normal):
        filename = f"normal_{i+1:03d}.png"
        filepath = f"images/{filename}"
        
        # Create synthetic normal X-ray
        img = create_synthetic_xray(is_abnormal=False, seed=42 + i)
        
        # Save image
        img_path = os.path.join(images_dir, filename)
        img.save(img_path, "PNG")
        
        # Add to labels
        labels_data.append({
            "filepath": filepath,
            "label": 0
        })
    
    # Create abnormal images
    for i in range(num_abnormal):
        filename = f"abnormal_{i+1:03d}.png"
        filepath = f"images/{filename}"
        
        # Create synthetic abnormal X-ray
        img = create_synthetic_xray(is_abnormal=True, seed=100 + i)
        
        # Save image
        img_path = os.path.join(images_dir, filename)
        img.save(img_path, "PNG")
        
        # Add to labels
        labels_data.append({
            "filepath": filepath,
            "label": 1
        })
    
    # Create and save labels CSV
    labels_df = pd.DataFrame(labels_data)
    labels_path = os.path.join(output_dir, "labels.csv")
    labels_df.to_csv(labels_path, index=False)
    
    print(f"Sample dataset created successfully!")
    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_path}")
    print(f"Dataset summary:")
    print(f"  Total images: {len(labels_df)}")
    print(f"  Normal images: {len(labels_df[labels_df['label'] == 0])}")
    print(f"  Abnormal images: {len(labels_df[labels_df['label'] == 1])}")
    
    return labels_df


def main():
    """Main function to create sample dataset."""
    parser = argparse.ArgumentParser(description="Generate synthetic sample data")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data/sample",
        help="Output directory for sample data"
    )
    parser.add_argument(
        "--num_normal", 
        type=int, 
        default=2,
        help="Number of normal images to create"
    )
    parser.add_argument(
        "--num_abnormal", 
        type=int, 
        default=2,
        help="Number of abnormal images to create"
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=320,
        help="Image size (square)"
    )
    
    args = parser.parse_args()
    
    # Create sample dataset
    labels_df = create_sample_dataset(
        output_dir=args.output_dir,
        num_normal=args.num_normal,
        num_abnormal=args.num_abnormal
    )
    
    print("\nSample dataset creation completed!")


if __name__ == "__main__":
    main()

