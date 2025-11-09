# NIH Chest X-ray Dataset

This directory contains the preprocessed NIH Chest X-ray dataset for the Medical X-ray Triage project.

## Dataset Information

- **Source**: [NIH Chest X-ray Dataset on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
- **Classes**: 
  - **Normal** (label=0): "No Finding" cases
  - **Pneumonia** (label=1): "Pneumonia" cases
- **Image Size**: 320×320 pixels (resized from original)
- **Splits**: 
  - Train: 70%
  - Validation: 15%
  - Test: 15%

## Directory Structure

After preprocessing, the dataset will have the following structure:

```
data/nih_chest_xray/
├── train/
│   ├── NORMAL/
│   │   └── *.png
│   └── PNEUMONIA/
│       └── *.png
├── val/
│   ├── NORMAL/
│   │   └── *.png
│   └── PNEUMONIA/
│       └── *.png
├── test/
│   ├── NORMAL/
│   │   └── *.png
│   └── PNEUMONIA/
│       └── *.png
├── train_labels.csv
├── val_labels.csv
├── test_labels.csv
├── labels.csv (copy of train_labels.csv for compatibility)
└── Data_Entry_2017.csv (original CSV file)
```

## Download and Setup

### Option 1: Using Kaggle API (Recommended)

1. **Install Kaggle package**:
   ```bash
   pip install kaggle
   ```

2. **Set up Kaggle API credentials**:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token" to download `kaggle.json`
   - Place `kaggle.json` in `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the dataset**:
   ```bash
   python scripts/download_nih_dataset.py --output_dir ./data/nih_chest_xray
   ```

   This will:
   - Download the dataset from Kaggle
   - Extract all zip files (images_001.zip to images_012.zip)
   - Extract Data_Entry_2017.csv

### Option 2: Manual Download

1. **Download from Kaggle**:
   - Go to https://www.kaggle.com/datasets/nih-chest-xrays/data
   - Click "Download" button
   - Extract all files to `./data/nih_chest_xray/`

2. **Verify structure**:
   - Ensure `Data_Entry_2017.csv` exists
   - Ensure images are extracted (either as individual files or in zip files)

## Preprocessing

After downloading the dataset, run the preprocessing script:

```bash
python src/preprocess_nih.py --data_dir ./data/nih_chest_xray
```

### Preprocessing Options

```bash
python src/preprocess_nih.py \
    --data_dir ./data/nih_chest_xray \
    --images_dir ./data/nih_chest_xray/images \  # Optional: specify images directory
    --output_dir ./data/nih_chest_xray \          # Optional: specify output directory
    --img_size 320 \                               # Image size (default: 320)
    --train_split 0.70 \                           # Training fraction (default: 0.70)
    --val_split 0.15 \                             # Validation fraction (default: 0.15)
    --test_split 0.15 \                            # Test fraction (default: 0.15)
    --seed 1337                                    # Random seed (default: 1337)
```

### What the Preprocessing Script Does

1. **Loads Data_Entry_2017.csv**: Reads the original CSV file
2. **Filters labels**: Keeps only "Pneumonia" and "No Finding" cases
3. **Creates labels**: 
   - "No Finding" → label=0 (Normal)
   - "Pneumonia" → label=1 (Pneumonia)
4. **Splits data**: Creates train/val/test splits (70/15/15) with stratification
5. **Resizes images**: Resizes all images to 320×320 pixels
6. **Organizes folders**: Creates train/, val/, test/ folders with NORMAL/ and PNEUMONIA/ subfolders
7. **Generates labels CSV**: Creates train_labels.csv, val_labels.csv, and test_labels.csv

## Usage

### Training

The training script will automatically detect the NIH dataset structure and use the pre-split data:

```bash
python -m src.train --data_dir ./data/nih_chest_xray --config config_example.yaml
```

Or update `config_example.yaml`:

```yaml
data_dir: "./data/nih_chest_xray"
```

### Evaluation

The evaluation script will automatically use the test split:

```bash
python -m src.eval --data_dir ./data/nih_chest_xray --model_path ./results/best.pt
```

### Using Specific Splits

You can also manually specify which split to use:

```bash
# Use validation split for evaluation
python -m src.eval \
    --data_dir ./data/nih_chest_xray \
    --labels_path ./data/nih_chest_xray/val_labels.csv \
    --images_dir ./data/nih_chest_xray/val \
    --model_path ./results/best.pt
```

## Labels CSV Format

The labels CSV files have the following format:

```csv
filepath,label
NORMAL/image_001.png,0
PNEUMONIA/image_002.png,1
...
```

- **filepath**: Relative path from the split directory (e.g., `NORMAL/image_001.png`)
- **label**: Binary label (0=Normal, 1=Pneumonia)

## Dataset Statistics

After preprocessing, you can check dataset statistics:

```python
import pandas as pd

# Check train split
train_df = pd.read_csv('./data/nih_chest_xray/train_labels.csv')
print(f"Train samples: {len(train_df)}")
print(f"Normal: {(train_df['label'] == 0).sum()}")
print(f"Pneumonia: {(train_df['label'] == 1).sum()}")

# Check val split
val_df = pd.read_csv('./data/nih_chest_xray/val_labels.csv')
print(f"Val samples: {len(val_df)}")

# Check test split
test_df = pd.read_csv('./data/nih_chest_xray/test_labels.csv')
print(f"Test samples: {len(test_df)}")
```

## Notes

- **Dataset Size**: The NIH Chest X-ray dataset is large (~112GB uncompressed). Ensure you have sufficient disk space.
- **Processing Time**: Preprocessing can take several hours depending on your system. The script uses progress bars to show progress.
- **Memory Requirements**: Resizing images requires sufficient RAM. If you encounter memory issues, consider processing in batches.
- **Class Imbalance**: The dataset may be imbalanced. The training script automatically handles class weights for balanced training.

## Troubleshooting

### Issue: "Data_Entry_2017.csv not found"

**Solution**: Ensure you've downloaded and extracted the dataset. Check that `Data_Entry_2017.csv` exists in `./data/nih_chest_xray/`.

### Issue: "Images directory not found"

**Solution**: Ensure images are extracted. The preprocessing script will search for images in subdirectories, but if images are in zip files, extract them first.

### Issue: "Out of memory" during preprocessing

**Solution**: 
- Process images in smaller batches (modify the script if needed)
- Use `--skip_resize` flag if images are already 320×320
- Ensure sufficient RAM is available

### Issue: "Missing images" warning

**Solution**: Some images in the CSV may not be present in the dataset. The preprocessing script will skip missing images and continue processing.

## References

- **NIH Chest X-ray Dataset**: https://www.kaggle.com/datasets/nih-chest-xrays/data
- **Original Paper**: Wang et al., "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases," *CVPR 2017*

