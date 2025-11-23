"""
Streamlit web interface for the Medical X-ray Triage project.

This application provides an interactive interface for uploading chest X-ray images,
running inference, and visualizing Grad-CAM results.

Refactored for improved UI/UX, clean code structure, and professional presentation.
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
import json
import time
from typing import Tuple, Optional, Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model import create_model, load_model
from data import get_transforms
from utils import get_device, seed_everything
from plotting import plot_gradcam_comparison
from uncertainty import monte_carlo_dropout_predict, compute_confidence_interval

# Try to import Grad-CAM
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False


# ============================================================================
# Configuration & Setup
# ============================================================================

st.set_page_config(
    page_title="Medical X-ray Triage",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'checkpoint_info' not in st.session_state:
    st.session_state.checkpoint_info = None


# ============================================================================
# Custom CSS Styling
# ============================================================================

CUSTOM_CSS = """
<style>
    /* Medical color scheme: whites, light grays, accent blue/teal */
    :root {
        --primary-blue: #1f77b4;
        --accent-teal: #17a2b8;
        --success-green: #28a745;
        --warning-orange: #fd7e14;
        --danger-red: #dc3545;
        --light-gray: #f8f9fa;
        --border-gray: #dee2e6;
        --text-dark: #262730;
    }
    
    .stApp {
        background-color: #ffffff;
        color: var(--text-dark);
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--primary-blue);
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-blue);
    }
    
    /* Disclaimer card */
    .disclaimer-card {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .disclaimer-card h4 {
        color: #856404;
        margin-top: 0;
        margin-bottom: 0.75rem;
    }
    
    .disclaimer-card p {
        color: #856404;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Result cards */
    .result-card {
        background-color: var(--light-gray);
        border: 1px solid var(--border-gray);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .result-card h3 {
        color: var(--primary-blue);
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.25rem;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.875rem;
        font-weight: 600;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    
    .badge-normal {
        background-color: var(--success-green);
        color: white;
    }
    
    .badge-abnormal {
        background-color: var(--danger-red);
        color: white;
    }
    
    .badge-risk-low {
        background-color: var(--success-green);
        color: white;
    }
    
    .badge-risk-medium {
        background-color: var(--warning-orange);
        color: white;
    }
    
    .badge-risk-high {
        background-color: var(--danger-red);
        color: white;
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-blue);
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-gray);
    }
    
    /* Info text */
    .info-text {
        color: #6c757d;
        font-size: 0.9rem;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    .sidebar-section {
        margin: 1.5rem 0;
        padding: 1rem;
        background-color: white;
        border-radius: 6px;
        border-left: 4px solid var(--primary-blue);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# Utility Functions
# ============================================================================

def load_optimal_threshold() -> float:
    """Load the optimal threshold from evaluation results if available."""
    eval_results_path = Path(__file__).parent.parent / "results" / "evaluation_results.json"
    if eval_results_path.exists():
        try:
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
            optimal_threshold = eval_results.get('optimal_threshold', 0.2580)
            return max(0.05, min(0.95, optimal_threshold))
        except (json.JSONDecodeError, KeyError):
            pass
    return 0.2580


@st.cache_resource
def load_trained_model(model_path: str, device: torch.device) -> Tuple[Optional[torch.nn.Module], Optional[Dict]]:
    """
    Load the trained model with caching.
    
    Args:
        model_path: Path to the model file
        device: Device to load the model on
    
    Returns:
        tuple: (model, checkpoint_info) or (None, None) if loading fails
    """
    if not os.path.exists(model_path):
        return None, None
    
    try:
        model, checkpoint_info = load_model(model_path, device)
        return model, checkpoint_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_image(image: Image.Image, img_size: int) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image: PIL Image
        img_size: Target image size
    
    Returns:
        Preprocessed tensor
    """
    transform = get_transforms(img_size=img_size, is_training=False)
    return transform(image).unsqueeze(0)


def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    use_uncertainty: bool = False,
    n_mc_samples: int = 50
) -> Dict:
    """
    Run model inference with optional uncertainty estimation.
    
    Args:
        model: Trained model
        input_tensor: Preprocessed input tensor
        device: Device to run on
        use_uncertainty: Whether to use Monte-Carlo dropout
        n_mc_samples: Number of MC samples if uncertainty enabled
    
    Returns:
        Dictionary with prediction results
    """
    input_tensor = input_tensor.to(device)
    model.eval()
    
    inference_start = time.time()
    
    with torch.no_grad():
        logits = model(input_tensor)
        prediction_prob = torch.sigmoid(logits)[0, 0].item()
    
    inference_time = (time.time() - inference_start) * 1000  # ms
    
    result = {
        'probability': prediction_prob,
        'inference_time': inference_time,
        'uncertainty_mean': prediction_prob,
        'uncertainty_std': 0.0,
        'confidence_lower': prediction_prob,
        'confidence_upper': prediction_prob
    }
    
    if use_uncertainty:
        try:
            uncertainty_mean, uncertainty_std, all_probs = monte_carlo_dropout_predict(
                model, input_tensor, n_samples=n_mc_samples
            )
            confidence_lower, confidence_upper = compute_confidence_interval(all_probs.flatten())
            result.update({
                'uncertainty_mean': uncertainty_mean,
                'uncertainty_std': uncertainty_std,
                'confidence_lower': confidence_lower,
                'confidence_upper': confidence_upper
            })
        except Exception as e:
            st.warning(f"Uncertainty estimation failed: {e}")
    
    return result


def get_target_layers(model: torch.nn.Module, model_name: str) -> List:
    """Get target layers for Grad-CAM."""
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith("resnet"):
        last_layer = model.backbone[-1]
        if hasattr(last_layer, '__getitem__'):
            last_block = last_layer[-1]
            if hasattr(last_block, 'conv2'):
                return [last_block.conv2]
        return [last_layer]
    elif model_name_lower.startswith("efficientnet"):
        return [model.backbone[-1]]
    else:
        return [model.backbone[-1]]


def compute_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    model_name: str,
    device: torch.device,
    cam_method: str = "GradCAM"
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Generate Grad-CAM visualization.
    
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
        # Ensure input_tensor is on the same device as the model
        model_device = next(model.parameters()).device
        input_tensor = input_tensor.to(model_device)
        
        model.eval()
        with torch.no_grad():
            logits = model(input_tensor)
            prediction_prob = torch.sigmoid(logits)[0, 0].item()
        
        target_layers = get_target_layers(model, model_name)
        out_dim = int(logits.shape[-1])
        target_idx = 0 if out_dim == 1 else 1
        
        if cam_method == "GradCAM":
            cam = GradCAM(model=model, target_layers=target_layers)
        elif cam_method == "GradCAMPlusPlus":
            cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        elif cam_method == "XGradCAM":
            cam = XGradCAM(model=model, target_layers=target_layers)
        else:
            return prediction_prob, None, None
        
        targets = [ClassifierOutputTarget(target_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]
        
        # Convert to numpy for overlay
        original_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        original_np = (original_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        original_np = np.clip(original_np, 0, 1)
        
        gradcam_image = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
        
        return prediction_prob, gradcam_image, grayscale_cam
    
    except Exception as e:
        st.warning(f"Grad-CAM generation failed: {e}")
        return None, None, None


def calculate_risk_level(probability: float, threshold: float) -> Tuple[str, str]:
    """
    Calculate risk level based on probability and threshold.
    
    Args:
        probability: Predicted probability
        threshold: Classification threshold
    
    Returns:
        tuple: (risk_level, risk_color_class)
    """
    if probability >= threshold:
        if probability > 0.8:
            return "High", "badge-risk-high"
        else:
            return "Medium", "badge-risk-medium"
    else:
        return "Low", "badge-risk-low"


# ============================================================================
# UI Rendering Functions
# ============================================================================

def render_header():
    """Render the main header and disclaimer."""
    st.markdown('<h1 class="main-header">🏥 Medical X-ray Triage System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="disclaimer-card">
        <h4>⚠️ Important Disclaimer</h4>
        <p><strong>This system is for research and educational purposes only.</strong></p>
        <p>It is <strong>NOT</strong> intended for clinical diagnosis or medical decision-making.</p>
        <p>Always consult qualified healthcare professionals for medical concerns.</p>
        <p><em>This is not a standalone medical device.</em></p>
    </div>
    """, unsafe_allow_html=True)
    

def render_sidebar() -> Dict:
    """Render sidebar configuration and return settings."""
    st.sidebar.title("⚙️ Configuration")
    
    # Model selection
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Model Selection")
    
    model_options = {
        "ResNet18": "./results/best.pt",
        "Custom Model": None
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        options=list(model_options.keys()),
        index=0,
        help="Choose the model to use for inference"
    )
    
    model_path = model_options[selected_model]
    
    if selected_model == "Custom Model":
        uploaded_model = st.sidebar.file_uploader(
            "Upload Model File",
            type=['pt', 'pth'],
            help="Upload a trained PyTorch model file"
        )
        if uploaded_model is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_file.write(uploaded_model.read())
                model_path = tmp_file.name
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Model parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Model Parameters")
    
    img_size = st.sidebar.slider(
        "Image Size",
        min_value=224,
        max_value=512,
        value=320,
        step=32,
        help="Size to resize images to (224, 256, 288, 320, 352, 384, 416, 448, 480, 512)"
    )
    
    optimal_threshold = load_optimal_threshold()
    threshold = st.sidebar.slider(
        "Classification Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(optimal_threshold),
        step=0.01,
        help=f"Probability threshold for abnormal classification. If probability ≥ threshold, classify as 'Abnormal'. Optimal: {optimal_threshold:.3f}"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Grad-CAM parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Grad-CAM Parameters")
    
    if GRADCAM_AVAILABLE:
        show_all_cam_methods = st.sidebar.checkbox(
            "Compare All Methods",
            value=False,
            help="Show side-by-side comparison of all Grad-CAM methods"
        )
        
        if not show_all_cam_methods:
            cam_method = st.sidebar.selectbox(
                "Grad-CAM Method",
                options=["GradCAM", "GradCAMPlusPlus", "XGradCAM"],
                index=0,
                help="Method for generating Grad-CAM visualizations"
            )
        else:
            cam_method = "All"
        
        show_gradcam = st.sidebar.checkbox(
            "Show Grad-CAM",
            value=True,
            help="Generate and display Grad-CAM visualizations"
        )
    else:
        st.sidebar.warning("Grad-CAM not available. Install with: pip install pytorch-grad-cam")
        show_gradcam = False
        cam_method = "GradCAM"
        show_all_cam_methods = False
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Uncertainty estimation
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Uncertainty Estimation")
    use_uncertainty = st.sidebar.checkbox(
        "Enable Uncertainty Estimation",
        value=False,
        help="Use Monte-Carlo dropout for uncertainty quantification"
    )
    
    if use_uncertainty:
        n_mc_samples = st.sidebar.slider(
            "MC Samples",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of Monte-Carlo samples for uncertainty estimation"
        )
    else:
        n_mc_samples = 50
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Device selection
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.subheader("Device")
    device_options = ["auto", "cpu", "cuda", "mps"]
    selected_device = st.sidebar.selectbox("Device", options=device_options, index=0)
    
    if selected_device == "auto":
        device = get_device()
    else:
        device = torch.device(selected_device)
    
    if device.type == "cpu":
        st.sidebar.info("Using CPU (slower). Consider using GPU if available.")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'model_path': model_path,
        'img_size': img_size,
        'threshold': threshold,
        'cam_method': cam_method,
        'show_gradcam': show_gradcam,
        'show_all_cam_methods': show_all_cam_methods,
        'use_uncertainty': use_uncertainty,
        'n_mc_samples': n_mc_samples,
        'device': device
    }


def render_prediction_card(result: Dict, threshold: float) -> None:
    """Render the prediction results card."""
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    prob = result['uncertainty_mean']
    pred_class = "Abnormal" if prob >= threshold else "Normal"
    badge_class = "badge-abnormal" if pred_class == "Abnormal" else "badge-normal"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h3>Prediction</h3>
            <p><strong>Class:</strong> <span class="badge {badge_class}">{pred_class}</span></p>
            <p><strong>Probability:</strong> {prob:.3f}</p>
            <p><strong>Threshold:</strong> {threshold:.3f}</p>
            <p class="info-text">If probability ≥ threshold, classify as 'Abnormal'</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if result.get('uncertainty_std', 0) > 0:
            st.markdown(f"""
            <div class="result-card">
                <h3>Uncertainty</h3>
                <p><strong>Mean:</strong> {result['uncertainty_mean']:.3f}</p>
                <p><strong>Std:</strong> {result['uncertainty_std']:.3f}</p>
                <p><strong>95% CI:</strong> [{result['confidence_lower']:.3f}, {result['confidence_upper']:.3f}]</p>
                <p class="info-text">Higher uncertainty indicates less confident predictions</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <h3>Performance</h3>
                <p><strong>Inference Time:</strong> {result['inference_time']:.2f} ms</p>
                <p><strong>Device:</strong> {result.get('device', 'N/A')}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Risk assessment
    risk_level, risk_badge = calculate_risk_level(prob, threshold)
    risk_explanation = {
        "High": "Model probability far above threshold - high confidence in abnormal classification",
        "Medium": "Model probability above threshold - moderate confidence",
        "Low": "Model probability below threshold - normal classification"
    }
    
    st.markdown(f"""
    <div class="result-card">
        <h3>Risk Assessment</h3>
        <p><strong>Risk Level:</strong> <span class="badge {risk_badge}">{risk_level}</span></p>
        <p>{risk_explanation[risk_level]}</p>
    </div>
    """, unsafe_allow_html=True)


def render_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    model_name: str,
    device: torch.device,
    cam_method: str,
    show_all: bool,
    img_size: int
) -> None:
    """Render Grad-CAM visualizations."""
    st.markdown('<div class="section-header">🎯 Grad-CAM Visualization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <p class="info-text">
    Grad-CAM highlights regions that most influenced the prediction. 
    This is for qualitative insight only and not a diagnostic tool.
    </p>
    """, unsafe_allow_html=True)
    
    if show_all and cam_method == "All":
        with st.spinner("Generating all Grad-CAM methods..."):
            gradcam_results = {}
            methods = ["GradCAM", "GradCAMPlusPlus", "XGradCAM"]
            
            for method in methods:
                try:
                    _, gradcam_img, grayscale_cam = compute_gradcam(
                        model, input_tensor, model_name, device, method
                    )
                    if gradcam_img is not None and grayscale_cam is not None:
                        gradcam_results[method] = (gradcam_img, grayscale_cam)
                except Exception as e:
                    st.warning(f"Failed to generate {method}: {e}")
            
            if gradcam_results:
                original_np = np.array(original_image.resize((img_size, img_size))) / 255.0
                fig = plot_gradcam_comparison(original_np, gradcam_results)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                st.info("Comparison of all Grad-CAM methods. Each method highlights different aspects of the model's attention.")
            else:
                st.warning("Could not generate any Grad-CAM visualizations.")
    else:
        with st.spinner("Generating Grad-CAM..."):
            pred_prob, gradcam_img, grayscale_cam = compute_gradcam(
                model, input_tensor, model_name, device, cam_method
            )
            
            if gradcam_img is not None and grayscale_cam is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(original_image, caption="Original X-ray")
                
                with col2:
                    st.image(gradcam_img, caption=f"Grad-CAM Overlay ({cam_method})")
                
                # Heatmap with colorbar
                fig, ax = plt.subplots(figsize=(5, 4), dpi=80)
                im = ax.imshow(grayscale_cam, cmap='jet', vmin=0, vmax=1)
                ax.set_title("Grad-CAM Heatmap (0 = Low, 1 = High Attention)", fontsize=12)
                ax.axis('off')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Attention Intensity', rotation=270, labelpad=15, fontsize=10)
                plt.tight_layout()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            else:
                st.warning("Could not generate Grad-CAM visualization.")


def render_model_info(checkpoint_info: Dict, device: torch.device, img_size: int, 
                     cam_method: str, use_uncertainty: bool, inference_time: float) -> None:
    """Render model information in a collapsible expander."""
    st.markdown('<div class="section-header">ℹ️ Model Information</div>', unsafe_allow_html=True)
    
    with st.expander("Model Details (for researchers)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Architecture:**", checkpoint_info.get('model_name', 'Unknown'))
            st.write("**Input Size:**", f"{img_size}x{img_size}")
            st.write("**Device:**", str(device))
            st.write("**Grad-CAM Method:**", cam_method if cam_method != "All" else "All Methods")
        
        with col2:
            st.write("**Uncertainty Estimation:**", "Enabled" if use_uncertainty else "Disabled")
            st.write("**Inference Latency:**", f"{inference_time:.2f} ms")
            
            # Enhanced runtime statistics
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                st.write("**System Resources:**")
                st.write(f"  • CPU Usage: {cpu_percent:.1f}%")
                st.write(f"  • Memory Usage: {memory.percent:.1f}% ({memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB)")
                
                # Device-specific info
                if device.type == "cuda":
                    try:
                        if torch.cuda.is_available():
                            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
                            st.write(f"  • GPU Memory: {gpu_allocated:.2f} GB / {gpu_memory:.2f} GB")
                    except:
                        pass
                elif device.type == "mps":
                    st.write(f"  • Device: Apple Silicon (MPS)")
            except ImportError:
                st.write("**System Resources:**", "psutil not available")
            
            if 'total_params' in checkpoint_info:
                st.write("**Parameters:**", f"{checkpoint_info['total_params']:,}")
            st.write("**Task:**", "Binary Classification (Normal/Abnormal)")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    render_header()
    
    # Get configuration from sidebar
    config = render_sidebar()
    
    # Main content area
    st.markdown("---")
    
    # Upload section
    st.markdown('<div class="section-header">📤 Upload X-ray Image</div>', unsafe_allow_html=True)
    
    upload_mode = st.radio(
        "Upload Mode",
        options=["Single Image", "Batch Upload"],
        horizontal=True,
        help="Choose single image or batch upload mode"
    )
    
    if upload_mode == "Single Image":
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image for analysis (PNG, JPG, JPEG)"
        )
        uploaded_files = [uploaded_file] if uploaded_file else []
    else:
        uploaded_files = st.file_uploader(
            "Choose X-ray images (multiple)",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload multiple chest X-ray images for batch analysis"
        )
        uploaded_file = uploaded_files[0] if uploaded_files else None
    
    # Process uploaded image(s)
    if uploaded_file is not None:
        try:
            # Validate file size (max 10MB)
            if hasattr(uploaded_file, 'size') and uploaded_file.size > 10 * 1024 * 1024:
                st.error("File too large. Please upload an image smaller than 10MB.")
                return
            
            image = Image.open(uploaded_file).convert('RGB')
            
            # Load model
            if config['model_path'] and os.path.exists(config['model_path']):
                with st.spinner("Loading model..."):
                    model, checkpoint_info = load_trained_model(config['model_path'], config['device'])
                
                if model is None:
                    st.error("Failed to load model. Please check the model file.")
                    return
                
                st.success(f"✅ Model loaded: {checkpoint_info.get('model_name', 'Unknown')}")
                
                # Preprocess image and move to device
                input_tensor = preprocess_image(image, config['img_size']).to(config['device'])
                
                # Run inference
                with st.spinner("Running inference..."):
                    result = run_inference(
                        model, input_tensor, config['device'],
                        config['use_uncertainty'], config['n_mc_samples']
                    )
                    result['device'] = str(config['device'])
                
                # Display results
                render_prediction_card(result, config['threshold'])
                
                # Grad-CAM visualization
                if config['show_gradcam'] and GRADCAM_AVAILABLE:
                    render_gradcam(
                        model, input_tensor, image,
                        checkpoint_info.get('model_name', 'Unknown'),
                        config['device'], config['cam_method'],
                        config['show_all_cam_methods'],
                        config['img_size']
                    )
                elif config['show_gradcam'] and not GRADCAM_AVAILABLE:
                    st.warning("Grad-CAM is not available. Install pytorch-grad-cam to enable visualizations.")
                
                # Model information
                render_model_info(
                    checkpoint_info, config['device'], config['img_size'],
                    config['cam_method'], config['use_uncertainty'],
                    result['inference_time']
                )
            
            else:
                st.warning("No model file found. Please train a model first or upload a custom model.")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.info("Please ensure you uploaded a valid image file (PNG, JPG, or JPEG).")
        
        else:
            # Show upload prompt
            st.info("👆 Upload an X-ray image to get started")
            
            # Display sample images if available
            sample_dir = Path(__file__).parent.parent / "data" / "sample" / "images"
            if sample_dir.exists():
                st.markdown("### 📁 Sample Images")
                sample_images = list(sample_dir.glob("*.png")) + list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.jpeg"))
                
                if sample_images:
                    cols = st.columns(min(len(sample_images), 4))
                    for i, img_path in enumerate(sample_images[:4]):
                        with cols[i]:
                            try:
                                sample_img = Image.open(img_path)
                                st.image(sample_img, caption=img_path.name)
                            except Exception:
                                pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem 0;">
        <p><strong>Medical X-ray Triage System</strong> | Research & Educational Use Only</p>
        <p>Built with PyTorch, Streamlit, and Grad-CAM</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    seed_everything(1337)
    main()
