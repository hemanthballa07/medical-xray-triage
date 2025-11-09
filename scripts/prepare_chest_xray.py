"""
Prepare chest-xray-pneumonia dataset: generate label CSVs and resize images.
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def prepare_dataset(data_dir: Path, img_size: int = 320):
    """Generate label CSVs and resize images."""
    print("=" * 60)
    print("Preparing Chest X-ray Pneumonia Dataset")
    print("=" * 60)
    
    splits = ['train', 'val', 'test']
    all_labels = {}
    
    for split in splits:
        print(f"\n📊 Processing {split} split...")
        split_dir = data_dir / split
        normal_dir = split_dir / "NORMAL"
        pneumonia_dir = split_dir / "PNEUMONIA"
        
        labels = []
        
        # Process NORMAL images (label = 0)
        if normal_dir.exists():
            normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg")) + list(normal_dir.glob("*.png"))
            print(f"   NORMAL: {len(normal_images)} images")
            
            for img_path in tqdm(normal_images, desc=f"      Processing NORMAL", leave=False):
                # Resize if needed
                try:
                    img = Image.open(img_path).convert('RGB')
                    if img.size != (img_size, img_size):
                        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                        img_resized.save(img_path, 'JPEG', quality=95)
                    
                    # Add to labels
                    filepath = os.path.join("NORMAL", img_path.name)
                    labels.append({
                        'filepath': filepath,
                        'label': 0
                    })
                except Exception as e:
                    print(f"      ⚠️  Error processing {img_path.name}: {e}")
                    continue
        
        # Process PNEUMONIA images (label = 1)
        if pneumonia_dir.exists():
            pneumonia_images = list(pneumonia_dir.glob("*.jpeg")) + list(pneumonia_dir.glob("*.jpg")) + list(pneumonia_dir.glob("*.png"))
            print(f"   PNEUMONIA: {len(pneumonia_images)} images")
            
            for img_path in tqdm(pneumonia_images, desc=f"      Processing PNEUMONIA", leave=False):
                # Resize if needed
                try:
                    img = Image.open(img_path).convert('RGB')
                    if img.size != (img_size, img_size):
                        img_resized = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
                        img_resized.save(img_path, 'JPEG', quality=95)
                    
                    # Add to labels
                    filepath = os.path.join("PNEUMONIA", img_path.name)
                    labels.append({
                        'filepath': filepath,
                        'label': 1
                    })
                except Exception as e:
                    print(f"      ⚠️  Error processing {img_path.name}: {e}")
                    continue
        
        # Save CSV
        if labels:
            df_labels = pd.DataFrame(labels)
            csv_path = data_dir / f"{split}_labels.csv"
            df_labels.to_csv(csv_path, index=False)
            all_labels[split] = df_labels
            
            normal_count = (df_labels['label'] == 0).sum()
            pneumonia_count = (df_labels['label'] == 1).sum()
            print(f"   ✅ {split}_labels.csv: {len(df_labels)} images")
            print(f"      - NORMAL (0): {normal_count}")
            print(f"      - PNEUMONIA (1): {pneumonia_count}")
    
    # Create unified labels.csv (copy of train for compatibility)
    if 'train' in all_labels:
        unified_csv = data_dir / "labels.csv"
        all_labels['train'].to_csv(unified_csv, index=False)
        print(f"\n   ✅ labels.csv: {len(all_labels['train'])} images")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    for split in splits:
        if split in all_labels:
            df = all_labels[split]
            print(f"{split.upper()}: {len(df)} images")
            print(f"  - NORMAL: {(df['label'] == 0).sum()}")
            print(f"  - PNEUMONIA: {(df['label'] == 1).sum()}")
    
    print("\n✅ Dataset preparation complete!")
    return all_labels

if __name__ == "__main__":
    data_dir = Path("./data/chest_xray")
    prepare_dataset(data_dir, img_size=320)

