#!/usr/bin/env python3
"""
Setup script for the Medical X-ray Triage project.

This script automates the initial setup process including:
- Environment verification
- Sample data generation
- Documentation generation
- Basic functionality testing
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is 3.10 or newer."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please upgrade to Python 3.10 or newer")
        return False


def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 
        'scikit-learn', 'matplotlib', 'seaborn', 
        'streamlit', 'pytorch-grad-cam', 'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All required packages are installed")
        return True


def generate_sample_data():
    """Generate sample data for testing."""
    return run_command(
        "python src/make_sample_data.py",
        "Generating sample data"
    )


def generate_documentation():
    """Generate documentation diagrams."""
    return run_command(
        "python docs/make_docs_art.py",
        "Generating documentation diagrams"
    )


def test_basic_functionality():
    """Test basic functionality of the system."""
    print("🧪 Testing basic functionality...")
    
    # Test data loading
    success1 = run_command(
        "python -c \"import sys; sys.path.append('src'); from data import print_dataset_info; print_dataset_info('data/sample/labels.csv', 'data/sample/images')\"",
        "Testing data loading",
        check=False
    )
    
    # Test model creation
    success2 = run_command(
        "python -c \"import sys; sys.path.append('src'); from model import create_model; model = create_model('resnet50'); print('Model created successfully')\"",
        "Testing model creation",
        check=False
    )
    
    # Test utility functions
    success3 = run_command(
        "python -c \"import sys; sys.path.append('src'); from utils import seed_everything; seed_everything(42); print('Utils working')\"",
        "Testing utility functions",
        check=False
    )
    
    return success1 and success2 and success3


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        'data/sample/images',
        'results',
        'docs',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")
    
    print("✅ All directories created")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print("\n📋 Next Steps:")
    print("1. 📚 Explore the setup notebook:")
    print("   jupyter notebook notebooks/setup.ipynb")
    
    print("\n2. 🏋️ Train a model:")
    print("   python src/train.py --epochs 5")
    
    print("\n3. 📊 Evaluate the model:")
    print("   python src/eval.py")
    
    print("\n4. 🎯 Generate Grad-CAM visualizations:")
    print("   python src/interpret.py")
    
    print("\n5. 🌐 Launch the web interface:")
    print("   streamlit run ui/app.py")
    
    print("\n📖 Documentation:")
    print("• README.md - Project overview and quick start")
    print("• data/README.md - Data format and usage")
    print("• reports/blueprint.md - Technical documentation")
    print("• docs/architecture.png - System architecture")
    print("• docs/wireframe.png - UI wireframe")
    
    print("\n⚠️  Remember: This system is for research and educational use only!")
    print("   Always consult qualified healthcare professionals for medical concerns.")
    
    print("\n🚀 Happy coding!")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup script for Medical X-ray Triage project")
    parser.add_argument("--skip-deps", action="store_true", 
                       help="Skip dependency checking")
    parser.add_argument("--skip-data", action="store_true", 
                       help="Skip sample data generation")
    parser.add_argument("--skip-docs", action="store_true", 
                       help="Skip documentation generation")
    parser.add_argument("--skip-tests", action="store_true", 
                       help="Skip functionality tests")
    
    args = parser.parse_args()
    
    print("🏥 Medical X-ray Triage Project Setup")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Checking Python version", check_python_version),
    ]
    
    if not args.skip_deps:
        steps.append(("Checking dependencies", check_dependencies))
    
    if not args.skip_data:
        steps.append(("Generating sample data", generate_sample_data))
    
    if not args.skip_docs:
        steps.append(("Generating documentation", generate_documentation))
    
    if not args.skip_tests:
        steps.append(("Testing functionality", test_basic_functionality))
    
    # Execute setup steps
    all_success = True
    for description, func in steps:
        print(f"\n{description}...")
        if not func():
            all_success = False
            break
    
    if all_success:
        print_next_steps()
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

