"""
Download NIH Chest X-ray dataset from Kaggle.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download kaggle.json
   - Place kaggle.json in ~/.kaggle/ (or set KAGGLE_CONFIG_DIR env variable)
   - Set permissions: chmod 600 ~/.kaggle/kaggle.json

Usage:
    python scripts/download_nih_dataset.py --output_dir ./data/nih_chest_xray
"""

import os
import argparse
from pathlib import Path
import zipfile
from tqdm import tqdm


def download_kaggle_dataset(dataset_name: str, output_dir: Path):
    """
    Download dataset from Kaggle using the Kaggle API.
    
    Args:
        dataset_name: Kaggle dataset name (e.g., "nih-chest-xrays/data")
        output_dir: Directory to save downloaded files
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "kaggle package not found. Install it with: pip install kaggle\n"
            "Then set up your Kaggle API credentials as described in the script docstring."
        )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    
    # Download dataset
    kaggle.api.dataset_download_files(
        dataset_name,
        path=str(output_dir),
        unzip=False
    )
    
    print("Download complete!")
    
    # Find and extract zip files
    zip_files = list(output_dir.glob("*.zip"))
    print(f"\nFound {len(zip_files)} zip files")
    
    # Extract all zip files
    for zip_file in tqdm(zip_files, desc="Extracting zip files"):
        print(f"Extracting: {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Optionally remove zip file after extraction
        # zip_file.unlink()
    
    print("\n✅ Dataset download and extraction complete!")
    print(f"\nFiles are in: {output_dir}")
    print("\nNext steps:")
    print(f"1. Verify Data_Entry_2017.csv exists in {output_dir}")
    print(f"2. Verify images are extracted (should have images_*.zip or images/ folder)")
    print(f"3. Run preprocessing: python src/preprocess_nih.py --data_dir {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download NIH Chest X-ray dataset from Kaggle"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/nih_chest_xray",
        help="Output directory for downloaded dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nih-chest-xrays/data",
        help="Kaggle dataset name"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).resolve()
    
    # Check if Kaggle credentials are set up
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("⚠️  Kaggle API credentials not found!")
        print(f"Expected: {kaggle_json}")
        print("\nTo set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token' to download kaggle.json")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, you can download manually from:")
        print("https://www.kaggle.com/datasets/nih-chest-xrays/data")
        return
    
    # Download dataset
    try:
        download_kaggle_dataset(args.dataset, output_dir)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nIf you encounter issues, you can download manually:")
        print("1. Go to https://www.kaggle.com/datasets/nih-chest-xrays/data")
        print("2. Click 'Download' button")
        print("3. Extract all files to:", output_dir)
        print("4. Run preprocessing: python src/preprocess_nih.py --data_dir", output_dir)


if __name__ == "__main__":
    main()

