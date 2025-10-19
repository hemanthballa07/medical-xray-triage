# Data Directory

This directory contains the data for the Medical X-ray Triage project.

## Structure

- `sample/` - Contains sample synthetic data for testing and demonstration
  - `images/` - Sample X-ray images (PNG format, 320x320 pixels)
  - `labels.csv` - Labels for the sample images with columns: `filepath`, `label`

## Sample Dataset

The sample dataset includes 4 synthetic chest X-ray images:

- 2 images labeled as normal (label=0)
- 2 images labeled as abnormal (label=1)

These images are programmatically generated using gradients and simple shapes to simulate chest X-rays for demonstration purposes.

## Using Real Data

To use real chest X-ray data:

1. **Prepare your dataset**:

   - Place images in a directory (e.g., `data/real/images/`)
   - Create a `labels.csv` file with columns: `filepath`, `label`
   - Ensure images are in PNG or JPG format
   - Recommended image size: 320x320 pixels or larger

2. **Update the CSV format**:

   ```csv
   filepath,label
   images/normal_001.png,0
   images/normal_002.png,0
   images/abnormal_001.png,1
   images/abnormal_002.png,1
   ```

3. **Update configuration**:
   - Modify `src/config.py` to point to your data directory
   - Or use command line arguments: `--data_dir ./data/real`

## Data Preprocessing

The system automatically applies the following preprocessing:

- Resize to specified dimensions (default: 320x320)
- Normalize using ImageNet statistics
- Convert to PyTorch tensors

## Dataset Splits

For training, the system uses stratified splits:

- Training: 60% of data
- Validation: 20% of data
- Testing: 20% of data

For the sample dataset (4 images), all images are used for training and testing.

## Class Imbalance

The system handles class imbalance through:

- Weighted random sampling
- Class weight adjustment in loss function
- Balanced metrics calculation

## Data Privacy and Ethics

- This is a research/educational project
- Use only publicly available or properly consented datasets
- Follow institutional review board (IRB) guidelines
- Ensure HIPAA compliance for real medical data
- This system is NOT for clinical use

## Recommended Datasets

For research purposes, consider these publicly available datasets:

- ChestX-ray8 (NIH)
- CheXpert (Stanford)
- MIMIC-CXR (MIT)
- PadChest (Valencia)

Always check dataset licenses and usage terms before downloading.

