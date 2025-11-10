"""
Streamlit web interface for the Medical X-ray Triage project.

This application provides an interactive interface for uploading chest X-ray images,
running inference, and visualizing Grad-CAM results.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import cv2
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import create_model, load_model
from data import get_transforms
from utils import get_device, seed_everything

# Try to import Grad-CAM
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


def load_optimal_threshold():
    """Load the optimal threshold from evaluation results if available."""
    eval_results_path = Path(__file__).parent.parent / "results" / "evaluation_results.json"
    if eval_results_path.exists():
        try:
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
            optimal_threshold = eval_results.get('optimal_threshold', 0.2580)
            # Clamp to reasonable range for slider
            return max(0.05, min(0.95, optimal_threshold))
        except (json.JSONDecodeError, KeyError):
            pass
    return 0.2580  # Default threshold from evaluation


# Page configuration
st.set_page_config(
    page_title="Medical X-ray Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Force light theme and ensure all text is visible
st.markdown("""
<style>
    /* Force light theme by overriding Streamlit's theme variables */
    :root {
        --primary-color: #1f77b4;
        --background-color: #ffffff;
        --secondary-background-color: #f0f2f6;
        --text-color: #262730;
        --font: "Source Sans Pro", sans-serif;
    }
    
    /* Force light background for entire app */
    .stApp {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Force light background for main content */
    .main {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Force light background for sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Force light background for block container */
    .block-container {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    /* Make text dark and visible - but exclude buttons */
    body, .main, section, div, p, span, label, h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    
    /* Buttons should have proper contrast - override the above */
    .stButton > button,
    button[data-baseweb="button"] {
        background-color: #1f77b4 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    .stButton > button:hover,
    button[data-baseweb="button"]:hover {
        background-color: #1a5f8f !important;
        color: #ffffff !important;
    }
    
    /* Dropdowns and selectboxes - Baseweb components */
    .stSelectbox > div > div,
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    .stSelectbox label,
    label[data-baseweb="label"] {
        color: #262730 !important;
    }
    
    /* Baseweb select value text */
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div {
        color: #262730 !important;
        background-color: #ffffff !important;
    }
    
    /* Dropdown options */
    ul[role="listbox"] li,
    div[data-baseweb="menu"] li {
        background-color: #ffffff !important;
        color: #262730 !important;
    }
    
    ul[role="listbox"] li:hover,
    div[data-baseweb="menu"] li:hover {
        background-color: #f0f2f6 !important;
        color: #262730 !important;
    }
    
    /* Sliders */
    .stSlider label {
        color: #262730 !important;
    }
    
    /* Headers and text */
    h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    
    p, span, div, label {
        color: #262730 !important;
    }
    
    /* Custom header */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4 !important;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Disclaimer box */
    .disclaimer {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 2rem;
        color: #262730 !important;
    }
    
    .disclaimer p,
    .disclaimer h4 {
        color: #262730 !important;
    }
    
    /* Metric card */
    .metric-card {
        background-color: #f8f9fa !important;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
        color: #262730 !important;
    }
    
    .metric-card p,
    .metric-card h3,
    .metric-card strong {
        color: #262730 !important;
    }
    
    /* Error message */
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24 !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #262730 !important;
    }
    
    /* Info boxes */
    .stInfo,
    .stWarning,
    .stError,
    .stSuccess {
        color: #262730 !important;
    }
    
    /* Make all images responsive - scale with browser zoom and container */
    .stImage img,
    img[data-testid="stImage"],
    [data-testid="stImage"] img,
    .element-container img {
        max-width: 100% !important;
        height: auto !important;
        width: auto !important;
        object-fit: contain !important;
        display: block;
    }
    
    /* Ensure image containers don't force fixed widths */
    [data-testid="stImage"],
    div[data-testid="stImage"] {
        max-width: 100% !important;
    }
    
    /* Make matplotlib/plotly containers scale with viewport */
    .stPlotlyChart,
    [data-testid="stPyplot"],
    .stPyplot {
        max-width: 100% !important;
        width: auto !important;
    }
    
    /* Prevent fixed pixel widths on image containers */
    div[data-testid="stImage"] > div {
        width: auto !important;
        max-width: 100% !important;
    }
    
    /* Top menu bar */
    header[data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    header[data-testid="stHeader"] * {
        color: #262730 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #262730 !important;
    }
    
    /* Checkbox labels */
    .stCheckbox label {
        color: #262730 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model(model_path, device):
    """
    Load the trained model with caching.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    if not os.path.exists(model_path):
        return None, None
    
    try:
        model, checkpoint_info = load_model(model_path, device)
        return model, checkpoint_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def get_target_layers(model, model_name):
    """Get target layers for Grad-CAM."""
    model_name_lower = model_name.lower()
    
    # For ResNet models, get the last convolutional layer in the last residual block
    if model_name_lower.startswith("resnet"):
        # The backbone is a Sequential, and the last layer is layer4 (for ResNet18/50)
        # Get the last BasicBlock/Bottleneck in layer4
        last_layer = model.backbone[-1]
        # For ResNet18/50, the last layer is a Sequential of blocks
        # Get the last block's conv2 layer (the actual conv layer we want)
        if hasattr(last_layer, '__getitem__'):
            last_block = last_layer[-1]
            # Get conv2 from the last block (this is the last convolutional layer)
            if hasattr(last_block, 'conv2'):
                target_layers = [last_block.conv2]
            else:
                # Fallback to the last layer
                target_layers = [last_layer]
        else:
            target_layers = [last_layer]
    elif model_name_lower.startswith("efficientnet"):
        # For EfficientNet, use the last feature layer
        target_layers = [model.backbone[-1]]
    else:
        # Default: use the last layer of backbone
        target_layers = [model.backbone[-1]]
    
    return target_layers


def generate_gradcam_streamlit(model, input_tensor, model_name, device, cam_method="GradCAM"):
    """
    Generate Grad-CAM visualization for Streamlit.
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        model_name: Name of the model
        device: Device to run inference on
        cam_method: Grad-CAM method to use
    
    Returns:
        tuple: (prediction_prob, gradcam_image, grayscale_cam)
    """
    if not GRADCAM_AVAILABLE:
        return None, None, None
    
    try:
        # Get prediction - compute logits and probability separately
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)  # shape [1, 1] for binary classification
            prediction_prob = torch.sigmoid(logits)[0, 0].item()
        
        # Get target layers from the model
        target_layers = get_target_layers(model, model_name)
        
        # Determine target index based on output dimension
        # For binary models with single logit output, use index 0
        out_dim = int(logits.shape[-1])
        target_idx = 0 if out_dim == 1 else 1  # If you ever switch to 2-logit head, 1 = "Abnormal"
        
        # Create Grad-CAM object - use model directly (pytorch-grad-cam works with PyTorch models)
        # Device handling is automatic based on where the model is placed
        if cam_method == "GradCAM":
            cam = GradCAM(model=model, target_layers=target_layers)
        elif cam_method == "GradCAMPlusPlus":
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        elif cam_method == "XGradCAM":
            cam = XGradCAM(model=model, target_layers=target_layers)
        else:
            return prediction_prob, None, None
        
        # Generate Grad-CAM with correct target index for binary model
        targets = [ClassifierOutputTarget(target_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # HxW (remove batch dimension)
        
        # Convert to numpy for overlay
        original_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        # Denormalize ImageNet normalization
        original_np = (original_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        original_np = np.clip(original_np, 0, 1)
        
        # Create Grad-CAM visualization
        gradcam_image = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
        
        return prediction_prob, gradcam_image, grayscale_cam
    
    except Exception as e:
        # Return None for Grad-CAM on error, but keep the prediction probability
        st.warning(f"Grad-CAM failed: {e}")
        return prediction_prob, None, None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical X-ray Triage System</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This system is for research and educational purposes only.</strong> 
        It is NOT intended for clinical diagnosis or medical decision-making. 
        Always consult qualified healthcare professionals for medical concerns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    model_options = {
        "ResNet18": "./results/best.pt",
        "Custom Model": None
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0
    )
    
    model_path = model_options[selected_model]
    
    # Custom model upload
    if selected_model == "Custom Model":
        uploaded_model = st.sidebar.file_uploader(
            "Upload Model File",
            type=['pt', 'pth'],
            help="Upload a trained PyTorch model file"
        )
        
        if uploaded_model is not None:
            # Save uploaded model temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    
    img_size = st.sidebar.slider(
        "Image Size",
        min_value=224,
        max_value=512,
        value=320,
        step=32,
        help="Size to resize images to"
    )
    
    # Load optimal threshold from evaluation results
    optimal_threshold = load_optimal_threshold()
    
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=optimal_threshold,
        step=0.01,
        help=f"Probability threshold for abnormal classification. Optimal threshold: {optimal_threshold:.6f}"
    )
    
    # Grad-CAM parameters
    if GRADCAM_AVAILABLE:
        st.sidebar.subheader("Grad-CAM Parameters")
        
        cam_method = st.sidebar.selectbox(
            "Grad-CAM Method",
            options=["GradCAM", "GradCAMPlusPlus", "XGradCAM"],
            index=0,
            help="Method for generating Grad-CAM visualizations"
        )
        
        show_gradcam = st.sidebar.checkbox(
            "Show Grad-CAM",
            value=True,
            help="Generate and display Grad-CAM visualizations"
        )
    else:
        st.sidebar.warning("Grad-CAM not available. Install with: pip install pytorch-grad-cam")
        show_gradcam = False
        cam_method = "GradCAM"
    
    # Device selection
    device_options = ["auto", "cpu", "cuda", "mps"]
    selected_device = st.sidebar.selectbox("Device", options=device_options, index=0)
    
    if selected_device == "auto":
        device = get_device()
    else:
        device = torch.device(selected_device)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
            
            # Load model
            if model_path and os.path.exists(model_path):
                with st.spinner("Loading model..."):
                    model, checkpoint_info = load_trained_model(model_path, device)
                
                if model is not None:
                    st.success(f"Model loaded: {checkpoint_info['model_name']}")
                    
                    # Preprocess image
                    transform = get_transforms(img_size=img_size, is_training=False)
                    input_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Run inference - compute logits and probability separately
                    with st.spinner("Running inference..."):
                        model.eval()
                        with torch.no_grad():
                            logits = model(input_tensor)  # shape [1, 1] for binary classification
                            prediction_prob = torch.sigmoid(logits)[0, 0].item()
                    
                    # Display results
                    with col2:
                        st.header("📊 Analysis Results")
                        
                        # Prediction
                        pred_class = "Abnormal" if prediction_prob > threshold else "Normal"
                        confidence = max(prediction_prob, 1 - prediction_prob)
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Prediction</h3>
                            <p><strong>Class:</strong> {pred_class}</p>
                            <p><strong>Probability:</strong> {prediction_prob:.3f}</p>
                            <p><strong>Confidence:</strong> {confidence:.3f}</p>
                            <p><strong>Threshold:</strong> {threshold:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk assessment
                        if prediction_prob > threshold:
                            risk_level = "High" if prediction_prob > 0.8 else "Medium"
                            risk_color = "#dc3545" if risk_level == "High" else "#fd7e14"
                        else:
                            risk_level = "Low"
                            risk_color = "#28a745"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>Risk Assessment</h3>
                            <p><strong>Risk Level:</strong> <span style="color: {risk_color};">{risk_level}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Grad-CAM visualization
                        if show_gradcam and GRADCAM_AVAILABLE:
                            st.subheader("🎯 Grad-CAM Visualization")
                            
                            with st.spinner("Generating Grad-CAM..."):
                                try:
                                    pred_prob, gradcam_img, grayscale_cam = generate_gradcam_streamlit(
                                        model, input_tensor, checkpoint_info['model_name'], 
                                        device, cam_method
                                    )
                                    
                                    if gradcam_img is not None and grayscale_cam is not None:
                                        # Display Grad-CAM overlay
                                        st.image(gradcam_img, caption=f"Grad-CAM Overlay ({cam_method})", use_column_width=True)
                                        
                                        # Display heatmap with responsive sizing
                                        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
                                        im = ax.imshow(grayscale_cam, cmap='jet')
                                        ax.set_title("Grad-CAM Heatmap")
                                        ax.axis('off')
                                        plt.colorbar(im, ax=ax)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.close(fig)  # Close figure to free memory
                                        
                                        st.info(f"Grad-CAM shows which regions the model focuses on for the '{pred_class}' prediction.")
                                    else:
                                        st.warning("Could not generate Grad-CAM visualization. Showing original image instead.")
                                        st.image(image, caption="Original Image", use_column_width=True)
                                except Exception as e:
                                    st.warning(f"Grad-CAM generation failed: {e}. Showing original image instead.")
                                    st.image(image, caption="Original Image", use_column_width=True)
                        
                        # Model information
                        st.subheader("ℹ️ Model Information")
                        st.write(f"**Model:** {checkpoint_info['model_name']}")
                        st.write(f"**Device:** {device}")
                        st.write(f"**Image Size:** {img_size}x{img_size}")
                        st.write(f"**Grad-CAM Method:** {cam_method}")
                
                else:
                    st.error("Failed to load model. Please check the model file.")
            
            else:
                st.warning("No model file found. Please train a model first or upload a custom model.")
        
        else:
            # Show sample images
            st.info("👆 Upload an X-ray image to get started")
            
            # Display sample images if available
            sample_dir = "./data/sample/images"
            if os.path.exists(sample_dir):
                st.subheader("📁 Sample Images")
                sample_images = [f for f in os.listdir(sample_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if sample_images:
                    cols = st.columns(min(len(sample_images), 4))
                    for i, img_name in enumerate(sample_images[:4]):
                        with cols[i]:
                            img_path = os.path.join(sample_dir, img_name)
                            sample_img = Image.open(img_path)
                            st.image(sample_img, caption=img_name, use_column_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Medical X-ray Triage System | Research & Educational Use Only</p>
        <p>Built with PyTorch, Streamlit, and Grad-CAM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(1337)
    
    # Run the app
    main()
