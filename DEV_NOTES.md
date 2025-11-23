# Developer / Reviewer Guide

This document provides quick reference commands and workflows for developers and reviewers working with the Medical X-ray Triage System.

## Quick Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/hemanthballa07/medical-xray-triage.git
cd medical-xray-triage

# Create conda environment
conda env create -f environment.yml
conda activate medxray

# Or use pip
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Run integrity tests
python test_integrity.py

# Expected output: All 27/27 tests passed
```

## Core Workflows

### Training Pipeline

```bash
# Train with default configuration
python -m src.train --config config_example.yaml

# Train with custom parameters
python -m src.train --data_dir ./data/chest_xray --epochs 25 --batch_size 8 --lr 0.0001

# Expected outputs:
# - results/best.pt (model checkpoint)
# - results/metrics.json (training metrics)
# - results/loss_curve.png (training history)
```

### Enhanced Evaluation

```bash
# Run comprehensive evaluation
python src/eval_enhanced.py --data_dir data/chest_xray --model_path results/best.pt

# Expected outputs:
# - results/evaluation_results.json (all metrics)
# - results/predictions.npz (raw predictions for plotting)
# - results/confusion_matrix_*.png (multiple thresholds)
# - results/roc_curve.png
# - results/calibration_curve.png
```

### Generate Additional Plots

```bash
# First ensure predictions.npz exists (from eval_enhanced.py)
# Then generate plots
python src/generate_additional_plots.py

# Expected outputs:
# - results/precision_recall_curve.png
# - results/roc_vs_threshold.png
# - results/f1_accuracy_vs_threshold.png
```

### Ablation Study

```bash
# Compare model architectures
python src/ablation_study.py --data_dir data/chest_xray --output_dir results/ablation

# Expected outputs:
# - results/ablation_study_results.csv
# - results/ablation/*.pt (model checkpoints)
# - results/ablation/*.json (metrics per model)
```

### Hyperparameter Optimization

```bash
# Run Optuna hyperparameter sweep
python src/hyperparameter_sweep.py --data_dir data/chest_xray --n_trials 50

# Expected outputs:
# - results/hyperparameter_sweep/
#   - best_params.json
#   - optimization_history.png
#   - study.db (Optuna database)
```

### Cross-Dataset Evaluation

```bash
# Evaluate on external dataset
python src/cross_dataset_eval.py --model_path results/best.pt --external_data_dir data/nih_chest_xray

# Expected outputs:
# - results/cross_dataset_results.json
# - results/cross_dataset_comparison.png
```

### Fairness Analysis

```bash
# Run subgroup audit
python src/audit_module.py --model_path results/best.pt --data_dir data/chest_xray

# Expected outputs:
# - results/audit_results.json
# - results/subgroup_metrics.png
```

## UI Development

### Launch Streamlit UI

```bash
# Start the web interface
streamlit run ui/app.py

# Access at http://localhost:8501
```

### UI Features to Test

1. **Single Image Upload**: Upload one X-ray image
2. **Batch Upload**: Upload multiple images simultaneously
3. **Grad-CAM Methods**: Toggle between GradCAM, GradCAM++, XGradCAM
4. **Uncertainty Toggle**: Enable/disable Monte-Carlo dropout
5. **Threshold Slider**: Adjust threshold and observe live metrics
6. **Model Transparency Panel**: View model info and runtime stats

## Docker Workflow

### Build and Run

```bash
# Build Docker image
docker build -t medical-xray-triage .

# Run with Docker Compose (recommended)
docker-compose up

# Or run directly
docker run -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  medical-xray-triage
```

### Verify Docker Build

```bash
# Check image
docker images | grep medical-xray-triage

# Test container
docker run --rm medical-xray-triage python test_integrity.py
```

## Output Directory Structure

```
results/
├── best.pt                          # Best model checkpoint
├── metrics.json                     # Training metrics
├── evaluation_results.json          # Comprehensive evaluation results
├── predictions.npz                   # Raw predictions (for plotting)
├── metadata.json                    # Reproducibility metadata
├── ablation_study_results.csv       # Ablation study comparison
├── *.png                            # Various plots and visualizations
├── ablation/                        # Ablation study outputs
│   ├── resnet18_best.pt
│   ├── resnet50_best.pt
│   └── efficientnet_v2_s_best.pt
└── hyperparameter_sweep/            # Optuna study outputs
    ├── best_params.json
    └── optimization_history.png
```

## Key Files and Their Purposes

### Source Code (`src/`)

- `train.py`: Main training script with early stopping and LR scheduling
- `eval_enhanced.py`: Comprehensive evaluation with multiple thresholds, calibration, error analysis
- `eval.py`: Standard evaluation script
- `interpret.py`: Grad-CAM visualization
- `ablation_study.py`: Model architecture comparison
- `hyperparameter_sweep.py`: Optuna-based hyperparameter optimization
- `cross_dataset_eval.py`: External dataset evaluation
- `audit_module.py`: Fairness and subgroup analysis
- `bootstrap_metrics.py`: Bootstrap confidence intervals
- `failure_analysis.py`: Error case visualization
- `uncertainty.py`: Monte-Carlo dropout implementation
- `plotting.py`: Additional plotting utilities

### Configuration

- `config_example.yaml`: Example training configuration
- `requirements.txt`: Python dependencies
- `environment.yml`: Conda environment specification
- `Dockerfile`: Docker container definition
- `docker-compose.yml`: Docker Compose configuration

### Documentation

- `README.md`: Main project documentation
- `DEPLOYMENT.md`: Deployment guide
- `DEV_NOTES.md`: This file
- `reports/deliverable3_report.tex`: IEEE-format LaTeX report
- `notebooks/deliverable3_evaluation.ipynb`: Evaluation notebook

## Testing and Verification

### Run All Tests

```bash
# Integrity check
python test_integrity.py
```

### Verify Model Loading

```bash
# Quick model load test
python -c "
from src.model import create_model
import torch
model = create_model('resnet18', num_classes=1, pretrained=True)
model.load_state_dict(torch.load('results/best.pt', map_location='cpu'))
print('✓ Model loaded successfully')
"
```

### Verify Data Loading

```bash
# Test data loader
python -c "
from src.data import create_pre_split_data_loaders
train_loader, val_loader, test_loader = create_pre_split_data_loaders(
    data_dir='data/chest_xray',
    batch_size=8,
    img_size=320
)
print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')
print(f'Test batches: {len(test_loader)}')
"
```

## Common Issues and Solutions

### Issue: CUDA out of memory

**Solution**: Reduce batch size
```bash
python -m src.train --batch_size 4  # Instead of 8
```

### Issue: ModuleNotFoundError

**Solution**: Install missing dependencies
```bash
pip install -r requirements.txt
```

### Issue: Data loading errors

**Solution**: Verify data structure
```bash
# Check data directory structure
ls -R data/chest_xray/
# Should have train/, val/, test/ subdirectories
```

### Issue: Docker build fails

**Solution**: Check Dockerfile and build context
```bash
# Build with verbose output
docker build -t medical-xray-triage . --progress=plain
```

## Performance Benchmarks

### Expected Training Time

- **ResNet18** (25 epochs): ~15-20 minutes on GPU, ~2-3 hours on CPU
- **ResNet50** (25 epochs): ~25-30 minutes on GPU, ~4-5 hours on CPU
- **EfficientNetV2-S** (25 epochs): ~20-25 minutes on GPU, ~3-4 hours on CPU

### Expected Inference Time

- **GPU (CUDA)**: ~12.3 ms per image
- **CPU**: ~45.2 ms per image
- **Batch (8 images, GPU)**: ~15.1 ms per image

### Memory Requirements

- **Training**: ~4-6 GB GPU memory (batch_size=8)
- **Inference**: ~2-3 GB GPU memory
- **Docker container**: ~3-4 GB disk space

## Reproducibility

### Fixed Random Seeds

All scripts use fixed random seeds (42) for reproducibility:
- Data splitting
- Model initialization
- Training process

### Metadata Tracking

Evaluation automatically saves metadata:
- System information (OS, Python version, PyTorch version)
- Training configuration
- Dataset information
- Git commit hash (if available)

## Contact

For questions or issues:
- **Author**: Hemanth Balla
- **Email**: hemanthballa1861@gmail.com
- **Repository**: https://github.com/hemanthballa07/medical-xray-triage

