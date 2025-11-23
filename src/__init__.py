"""
Medical X-ray Triage with CNNs, Grad-CAM, and Streamlit UI

This package provides tools for binary abnormality detection in chest X-rays
using pretrained convolutional neural networks and interpretability techniques.
"""

__version__ = "1.0.0"
__author__ = "hemanthballa"
__email__ = "hemanthballa1861@gmail.com"

# Import main modules
from . import config
from . import data
from . import model
from . import train
from . import eval
from . import interpret
from . import utils

__all__ = [
    "config",
    "data", 
    "model",
    "train",
    "eval",
    "interpret",
    "utils"
]


