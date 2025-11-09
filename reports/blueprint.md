---
title: "Medical X-ray Triage System - Technical Blueprint"
author: "Hemanth Balla"
date: "October 19, 2025"
---

## Executive Summary

This document outlines the technical architecture and implementation details for a Medical X-ray Triage system that uses deep learning to detect abnormalities in chest X-ray images. The system combines convolutional neural networks (CNNs), Grad-CAM interpretability, and a Streamlit web interface to provide an end-to-end solution for medical image analysis.

**⚠️ Important Disclaimer**: This system is designed for research and educational purposes only. It is NOT intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical concerns.

## 1. System Overview

### 1.1 Objectives

- **Primary Goal**: Binary classification of chest X-ray images (Normal vs. Abnormal)
- **Secondary Goal**: Provide interpretable AI explanations through Grad-CAM visualizations
- **Tertiary Goal**: Deliver an intuitive web interface for easy interaction

### 1.2 Key Features

- Pretrained CNN models (ResNet50, EfficientNetV2-S)
- Grad-CAM interpretability for model transparency
- Interactive Streamlit web interface
- Comprehensive evaluation metrics
- Reproducible training pipeline
- Sample dataset for immediate testing

### 1.3 Target Metrics

- **AUROC**: > 0.95 (Area Under ROC Curve)
- **F1-Score**: > 0.90
- **Sensitivity**: > 0.90 (True Positive Rate)
- **Specificity**: > 0.90 (True Negative Rate)

## 2. Technical Architecture

### 2.1 System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │ Training Layer  │
│                 │    │                 │    │                 │
│ • Chest X-rays  │───▶│ • ResNet50      │───▶│ • BCE Loss      │
│ • Labels        │    │ • EfficientNet  │    │ • Adam Optimizer│
│ • Augmentation  │    │ • Custom Head   │    │ • Early Stop    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Inference Layer │    │Interpretability │    │  Metrics Layer  │
│                 │    │     Layer       │    │                 │
│ • Binary Class  │    │ • Grad-CAM      │    │ • AUROC         │
│ • Probability   │    │ • Heatmaps      │    │ • F1-Score      │
│ • Risk Assess   │    │ • Explanations  │    │ • Confusion Mat │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   UI Layer      │
                       │                 │
                       │ • Streamlit App │
                       │ • Image Upload  │
                       │ • Visualization │
                       └─────────────────┘
```

### 2.2 Data Flow

1. **Input**: Chest X-ray images (PNG/JPG, 320x320 pixels)
2. **Preprocessing**: Resize, normalize, apply transforms
3. **Model Inference**: Forward pass through pretrained CNN
4. **Postprocessing**: Sigmoid activation, threshold comparison
5. **Interpretability**: Grad-CAM visualization generation
6. **Output**: Classification result + confidence + heatmap

### 2.3 Technology Stack

| Component            | Technology          | Version | Purpose                 |
| -------------------- | ------------------- | ------- | ----------------------- |
| **Framework**        | PyTorch             | 2.0+    | Deep learning framework |
| **Computer Vision**  | Torchvision         | 0.15+   | Pretrained models       |
| **Interpretability** | pytorch-grad-cam    | 1.4+    | Grad-CAM visualizations |
| **Web Interface**    | Streamlit           | 1.28+   | Interactive UI          |
| **Data Processing**  | NumPy, Pandas       | Latest  | Data manipulation       |
| **Visualization**    | Matplotlib, Seaborn | Latest  | Plotting and metrics    |
| **ML Metrics**       | Scikit-learn        | 1.3+    | Evaluation metrics      |

## 3. Model Architecture

### 3.1 Backbone Models

#### ResNet50

- **Input**: 320x320x3 RGB images
- **Architecture**: ResNet50 with ImageNet pretrained weights
- **Feature Extraction**: 2048-dimensional feature vector
- **Custom Head**: 2-layer MLP (2048 → 512 → 1)
- **Parameters**: ~25M total, ~2M trainable

#### EfficientNetV2-S

- **Input**: 320x320x3 RGB images
- **Architecture**: EfficientNetV2-S with ImageNet pretrained weights
- **Feature Extraction**: 1280-dimensional feature vector
- **Custom Head**: 2-layer MLP (1280 → 512 → 1)
- **Parameters**: ~22M total, ~1.5M trainable

### 3.2 Training Configuration

| Parameter         | Value             | Description                        |
| ----------------- | ----------------- | ---------------------------------- |
| **Batch Size**    | 8                 | Training batch size                |
| **Learning Rate** | 1e-4              | Adam optimizer learning rate       |
| **Weight Decay**  | 1e-4              | L2 regularization                  |
| **Epochs**        | 2-10              | Training epochs (configurable)     |
| **Image Size**    | 320x320           | Input image dimensions             |
| **Loss Function** | BCEWithLogitsLoss | Binary cross-entropy with logits   |
| **Optimizer**     | AdamW             | Adaptive moment estimation         |
| **Scheduler**     | ReduceLROnPlateau | Learning rate reduction on plateau |

### 3.3 Data Augmentation

| Augmentation               | Parameters                   | Purpose                    |
| -------------------------- | ---------------------------- | -------------------------- |
| **Random Horizontal Flip** | p=0.5                        | Increase dataset diversity |
| **Random Rotation**        | degrees=5                    | Handle image orientation   |
| **Color Jitter**           | brightness=0.1, contrast=0.1 | Robustness to lighting     |
| **Normalization**          | ImageNet stats               | Standard preprocessing     |

This modular design enables efficient experimentation and ensures transparency across all stages of inference.

## 4. Implementation Details

### 4.1 Project Structure

```
pneumonia-project/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment
├── data/                     # Data directory
│   ├── sample/              # Sample dataset
│   │   ├── images/          # X-ray images
│   │   └── labels.csv       # Image labels
│   └── README.md            # Data documentation
├── src/                     # Source code
│   ├── config.py            # Configuration management
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # Model definitions
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   ├── interpret.py         # Grad-CAM interpretation
│   ├── utils.py             # Utility functions
│   └── make_sample_data.py  # Sample data generation
├── ui/                      # User interface
│   └── app.py               # Streamlit application
├── notebooks/               # Jupyter notebooks
│   └── setup.ipynb         # Environment setup
├── docs/                    # Documentation
│   ├── architecture.png     # System architecture diagram
│   └── wireframe.png        # UI wireframe
├── results/                 # Training outputs
└── reports/                 # Reports
    ├── blueprint.md         # This document
    └── blueprint.pdf        # PDF version
```

### 4.2 Key Classes and Functions

#### Data Pipeline

- `ChestXrayDataset`: Custom PyTorch dataset for X-ray images
- `get_transforms()`: Data augmentation and preprocessing
- `create_data_loaders()`: Train/val/test split with weighted sampling

#### Model Architecture

- `ChestXrayClassifier`: Main model wrapper with custom head
- `create_model()`: Factory function for model creation
- `load_model()`: Model checkpoint loading

#### Training Pipeline

- `train_epoch()`: Single epoch training loop
- `validate_epoch()`: Validation loop with metrics
- `EarlyStopping`: Prevents overfitting

#### Evaluation

- `compute_metrics()`: Comprehensive metric calculation
- `plot_roc_curve()`: ROC curve visualization
- `plot_confusion_matrix()`: Confusion matrix plot

#### Interpretability

- `generate_gradcam()`: Grad-CAM visualization generation
- `ChestXrayGradCAMWrapper`: Model wrapper for pytorch-grad-cam

## 5. User Interface Design

### 5.1 Streamlit Application Features

#### Sidebar Configuration

- **Model Selection**: Choose between ResNet50 and EfficientNetV2-S
- **Custom Model Upload**: Support for user-trained models
- **Image Size Slider**: Adjustable input dimensions (224-512)
- **Classification Threshold**: Probability threshold tuning (0.0-1.0)
- **Grad-CAM Method**: Select visualization method
- **Device Selection**: CPU/CUDA/MPS support

#### Main Interface

- **Image Upload**: Drag-and-drop or file browser
- **Image Preview**: Real-time image display
- **Analysis Results**:
  - Binary classification (Normal/Abnormal)
  - Confidence score
  - Risk level assessment
- **Grad-CAM Visualization**:
  - Heatmap overlay
  - Raw heatmap
  - Model explanation

#### Additional Features

- **Sample Images**: Built-in example images
- **Model Information**: Display model metadata
- **Disclaimer**: Clear research-only notice

### 5.2 UI Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    🏥 Medical X-ray Triage System           │
├─────────────────────────────────────────────────────────────┤
│ ⚠️  For Research and Educational Use Only                   │
├─────────────┬───────────────────────────────────────────────┤
│ Sidebar     │ Main Content Area                             │
│             │                                               │
│ Model       │ 📤 Upload X-ray Image                        │
│ Selection   │                                               │
│             │ ┌─────────────────────────────────────────┐   │
│ Image Size  │ │                                         │   │
│ Slider      │ │        Image Preview Area               │   │
│             │ │                                         │   │
│ Threshold   │ │                                         │   │
│ Slider      │ └─────────────────────────────────────────┘   │
│             │                                               │
│ Grad-CAM    │ 📊 Analysis Results                          │
│ Method      │ • Prediction: Normal/Abnormal               │
│             │ • Confidence: 0.XXX                         │
│ Device      │ • Risk Level: Low/Medium/High               │
│ Selection   │                                               │
│             │ 🎯 Grad-CAM Visualization                    │
│             │ • Heatmap Overlay                           │
│             │ • Raw Heatmap                               │
└─────────────┴───────────────────────────────────────────────┘
```

## 6. Evaluation and Metrics

### 6.1 Performance Metrics

#### Classification Metrics

- **AUROC**: Area Under ROC Curve (primary metric)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall (Sensitivity)**: True Positives / (True Positives + False Negatives)
- **Specificity**: True Negatives / (True Negatives + False Positives)

#### Additional Metrics

- **Confusion Matrix**: Detailed classification breakdown
- **ROC Curve**: Receiver Operating Characteristic curve
- **Optimal Threshold**: Youden's J statistic for threshold selection

### 6.2 Model Evaluation Process

1. **Train/Validation/Test Split**: 60/20/20 stratified split
2. **Cross-Validation**: 5-fold cross-validation for robust evaluation
3. **Threshold Optimization**: Find optimal classification threshold
4. **Statistical Significance**: Confidence intervals for metrics
5. **Baseline Comparison**: Compare against random classifier

### 6.3 Sample Dataset Performance

| Metric      | Target | Expected |
| ----------- | ------ | -------- |
| AUROC       | > 0.95 | ~0.98    |
| F1-Score    | > 0.90 | ~0.95    |
| Sensitivity | > 0.90 | ~0.95    |
| Specificity | > 0.90 | ~0.95    |

## 7. Interpretability and Explainability

### 7.1 Grad-CAM Implementation

#### Method Selection

- **Grad-CAM**: Standard gradient-weighted class activation mapping
- **Grad-CAM++**: Enhanced version with better localization
- **XGrad-CAM**: Improved gradient computation

#### Target Layers

- **ResNet50**: `layer4` (last convolutional layer)
- **EfficientNetV2-S**: `features.7` (last feature layer)

#### Visualization Output

- **Heatmap**: Red regions indicate high attention
- **Overlay**: Superimposed on original image
- **Raw Heatmap**: Pure attention map

### 7.2 Interpretation Guidelines

#### Normal Images

- **Expected**: Attention on lung fields, heart, and chest structure
- **Unexpected**: Concentrated attention on single regions

#### Abnormal Images

- **Expected**: Attention on pathological regions (opacities, consolidations)
- **Unexpected**: Attention on normal anatomical structures

## 8. Deployment and Usage

### 8.1 Installation Process

#### Prerequisites

- Python 3.10 or newer
- CUDA-capable GPU (optional, for faster training)
- 8GB+ RAM recommended

#### Setup Commands

```bash
# Clone repository
git clone <repository-url>
cd pneumonia-project

# Create environment
conda env create -f environment.yml
conda activate medxray

# Generate sample data
python src/make_sample_data.py

# Verify setup
jupyter notebook notebooks/setup.ipynb
```

### 8.2 Usage Workflow

#### Training

```bash
# Train ResNet50 model
python src/train.py --model_name resnet50 --epochs 5

# Train EfficientNetV2-S model
python src/train.py --model_name efficientnet_v2_s --epochs 5
```

#### Evaluation

```bash
# Evaluate trained model
python src/eval.py --model_path results/best.pt

# Generate Grad-CAM visualizations
python src/interpret.py --cam_method GradCAM
```

#### Web Interface

```bash
# Launch Streamlit app
streamlit run ui/app.py
```

### 8.3 Customization Options

#### Model Configuration

- **Backbone**: ResNet50, EfficientNetV2-S, or custom
- **Image Size**: 224, 320, 384, 512 pixels
- **Learning Rate**: 1e-5 to 1e-3
- **Batch Size**: 4, 8, 16, 32

#### Data Configuration

- **Custom Dataset**: Replace sample data with real X-rays
- **Augmentation**: Adjust augmentation parameters
- **Class Weights**: Handle imbalanced datasets

## 9. Ethical and Responsible AI Considerations

### 9.1 Technical Limitations

#### Model Limitations

- **Pretrained on Natural Images**: May not capture medical image nuances
- **Binary Classification**: Cannot distinguish between different pathologies
- **Single View**: Only handles frontal chest X-rays
- **Resolution**: Limited to 320x320 input resolution

#### Data Limitations

- **Sample Dataset**: Synthetic data may not reflect real-world performance
- **Limited Diversity**: Small dataset may not cover all variations
- **No Ground Truth**: Synthetic abnormalities may not match real pathology

### 9.2 Clinical Limitations

#### Diagnostic Accuracy

- **Research Tool**: Not validated for clinical use
- **False Positives/Negatives**: May miss or misclassify pathologies
- **No Clinical Context**: Lacks patient history and symptoms
- **Single Modality**: Only considers X-ray images

#### Regulatory Considerations

- **FDA Approval**: Not approved for clinical diagnosis
- **Medical Device**: May require regulatory approval for clinical use
- **Liability**: Users assume full responsibility for interpretation

### 9.3 Ethical Considerations

#### Bias and Fairness

- **Training Data Bias**: May reflect biases in training data
- **Demographic Bias**: Performance may vary across populations
- **Representation**: Ensure diverse representation in training data

#### Privacy and Security

- **Patient Data**: Handle medical images with appropriate privacy measures
- **HIPAA Compliance**: Follow healthcare privacy regulations
- **Data Retention**: Implement appropriate data retention policies

## 10. Implementation Timeline

### 10.1 Technical Improvements

#### Model Enhancements

- **Multi-class Classification**: Distinguish between different pathologies
- **Attention Mechanisms**: Improve model interpretability
- **Ensemble Methods**: Combine multiple models for better performance
- **Transfer Learning**: Fine-tune on medical imaging datasets

#### Data Improvements

- **Real Medical Data**: Incorporate actual chest X-ray datasets
- **Data Augmentation**: Advanced augmentation techniques
- **Multi-view Support**: Handle lateral and oblique views
- **High Resolution**: Support for higher resolution images

### 10.2 Clinical Integration

#### Validation Studies

- **Clinical Validation**: Validate on real patient data
- **Radiologist Comparison**: Compare with expert radiologists
- **Multi-center Studies**: Validate across different institutions
- **Longitudinal Studies**: Track performance over time

#### Integration Features

- **DICOM Support**: Handle standard medical imaging format
- **PACS Integration**: Connect with Picture Archiving systems
- **Workflow Integration**: Integrate with clinical workflows
- **Reporting**: Generate standardized reports

### 10.3 User Experience

#### Interface Improvements

- **Mobile Support**: Responsive design for mobile devices
- **Batch Processing**: Handle multiple images simultaneously
- **Export Features**: Export results and visualizations
- **User Management**: Multi-user support with authentication

#### Advanced Features

- **Real-time Processing**: Live image processing capabilities
- **Cloud Deployment**: Scalable cloud-based deployment
- **API Integration**: RESTful API for third-party integration
- **Analytics Dashboard**: Usage analytics and performance monitoring

## 11. Implementation Timeline

### Project Schedule

The following timeline outlines the development phases for the Medical X-ray Triage system:

| **Week**       | **Phase**                     | **Key Activities**                                                 |
| -------------- | ----------------------------- | ------------------------------------------------------------------ |
| Oct 20 – 26    | Data cleaning, baseline setup | Dataset preparation, environment setup, initial model architecture |
| Oct 27 – Nov 2 | Baseline training, simple UI  | Model training pipeline, basic Streamlit interface                 |
| Nov 3 – 16     | Tuning and interpretability   | Hyperparameter optimization, Grad-CAM integration                  |
| Nov 17 – 30    | UI integration and refinement | Advanced UI features, testing, documentation                       |
| Dec 1 – 11     | Demo and final report         | Final testing, deployment, comprehensive documentation             |

### Milestone Deliverables

- **Week 2**: Working training pipeline with ResNet18/ResNet50
- **Week 4**: Functional Grad-CAM visualization system
- **Week 6**: Complete Streamlit web interface
- **Week 8**: Comprehensive documentation and testing

## 12. Limitations and Considerations

### 12.1 Dataset Limitations

#### Small Sample Size

- **Current Demo Dataset**: Contains only 4 images (2 normal, 2 abnormal)
- **Impact**: Metrics may appear artificially perfect due to extreme class separation
- **Example**: AUROC = 1.0 with optimal threshold ≈1.22e-14, but default threshold (0.5) classifies all as Normal
- **Recommendation**: Use larger, balanced datasets for meaningful performance evaluation

#### Synthetic Data

- **Current Implementation**: Uses programmatically generated synthetic images
- **Limitation**: May not capture real-world complexity and variability
- **Clinical Relevance**: Results are illustrative only, not clinically validated

### 12.2 Model Limitations

#### Transfer Learning Constraints

- **Base Model**: Pre-trained on natural images (ImageNet)
- **Domain Gap**: Natural images vs. medical X-rays have different visual characteristics
- **Impact**: May require more training data or domain-specific pre-training

#### Binary Classification

- **Current Scope**: Only distinguishes Normal vs. Abnormal
- **Missing**: Specific pathology identification (pneumonia, COVID-19, etc.)
- **Extension**: Requires multi-class classification architecture

### 12.3 Technical Limitations

#### Hardware Requirements

- **Training**: Requires GPU for efficient training on large datasets
- **Inference**: CPU inference may be slow for real-time applications
- **Memory**: Large models require significant RAM

#### Reproducibility

- **Dependencies**: Specific package versions required for exact reproduction
- **Hardware**: Results may vary across different GPU/CPU architectures
- **Randomness**: Despite fixed seeds, some operations may still be non-deterministic

### 12.4 Clinical Considerations

#### Not for Clinical Use

- **Research Only**: System is designed for educational and research purposes
- **No Clinical Validation**: Has not been validated against real clinical outcomes
- **Regulatory Compliance**: Does not meet medical device standards

#### Interpretability Limitations

- **Grad-CAM**: Shows "where" the model looks, not "why"
- **False Interpretations**: Heatmaps may highlight irrelevant regions
- **Expert Review**: Requires radiologist validation for clinical insights

## 13. Conclusion

The Medical X-ray Triage system represents a comprehensive approach to automated chest X-ray analysis using state-of-the-art deep learning techniques. The system combines robust model architecture, interpretable AI methods, and an intuitive user interface to provide a complete solution for research and educational purposes.

### Key Achievements

- ✅ **Complete Implementation**: End-to-end pipeline from data to visualization
- ✅ **Interpretable AI**: Grad-CAM integration for model transparency
- ✅ **User-Friendly Interface**: Intuitive Streamlit web application
- ✅ **Reproducible Results**: Deterministic training with fixed seeds
- ✅ **Comprehensive Evaluation**: Multiple metrics and visualizations
- ✅ **Documentation**: Complete technical documentation and user guides

### Impact and Applications

- **Education**: Medical students and residents can learn AI-assisted diagnosis
- **Research**: Researchers can build upon this foundation for advanced studies
- **Development**: Software developers can understand medical AI implementation
- **Validation**: Provides a baseline for comparing different approaches

### Next Steps

1. **Data Collection**: Gather real chest X-ray datasets for validation
2. **Clinical Validation**: Conduct studies with radiologists
3. **Performance Optimization**: Improve model accuracy and speed
4. **Feature Enhancement**: Add multi-class classification capabilities
5. **Deployment**: Scale for production use with appropriate safeguards

This system serves as a foundation for advancing medical AI research while maintaining strict adherence to research-only usage guidelines. The combination of technical rigor, interpretability, and user accessibility makes it a valuable tool for the medical AI community.

## References

- Selvaraju, R. R., et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." _International Conference on Computer Vision (ICCV)_, 2017.
- He, K., et al. "Deep Residual Learning for Image Recognition." _Conference on Computer Vision and Pattern Recognition (CVPR)_, 2016.
- Tan, M., & Le, Q. "EfficientNetV2: Smaller Models and Faster Training." _International Conference on Machine Learning (ICML)_, 2021.
- Paszke, A., et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." _Advances in Neural Information Processing Systems (NeurIPS)_, 2019.

---

**Document Version**: 1.0  
**Last Updated**: October 19, 2025  
**Author**: Hemanth Balla  
**Review Status**: Ready for Submission
