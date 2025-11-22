"""
Test script to verify all Deliverable 3 requirements are met.

This script checks:
1. All required modules exist
2. All required functions are present
3. All required outputs are generated
4. Scripts can be imported without errors
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))


def check_module_exists(module_name, description):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {description}: {module_name} - {e}")
        return False


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} - NOT FOUND")
        return False


def check_function_exists(module, function_name, description):
    """Check if a function exists in a module."""
    try:
        func = getattr(module, function_name)
        if callable(func):
            print(f"✓ {description}: {function_name}")
            return True
        else:
            print(f"✗ {description}: {function_name} - Not callable")
            return False
    except AttributeError:
        print(f"✗ {description}: {function_name} - NOT FOUND")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Deliverable 3 Requirements Check")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Check required modules
    print("\n1. Checking Required Modules:")
    print("-" * 60)
    
    modules_to_check = [
        ("src.uncertainty", "Uncertainty estimation module"),
        ("src.plotting", "Plotting utilities module"),
        ("src.ablation_study", "Ablation study module"),
        ("src.generate_additional_plots", "Additional plots generator"),
        ("src.eval_enhanced", "Enhanced evaluation module"),
    ]
    
    for module_name, description in modules_to_check:
        if not check_module_exists(module_name, description):
            all_passed = False
    
    # 2. Check required functions in uncertainty.py
    print("\n2. Checking Uncertainty Module Functions:")
    print("-" * 60)
    try:
        import src.uncertainty as uncertainty
        functions = [
            ("monte_carlo_dropout_predict", "Monte-Carlo dropout prediction"),
            ("compute_confidence_interval", "Confidence interval computation"),
        ]
        for func_name, desc in functions:
            if not check_function_exists(uncertainty, func_name, desc):
                all_passed = False
    except ImportError:
        all_passed = False
    
    # 3. Check required functions in plotting.py
    print("\n3. Checking Plotting Module Functions:")
    print("-" * 60)
    try:
        import src.plotting as plotting
        functions = [
            ("plot_precision_recall_curve", "Precision-Recall curve"),
            ("plot_roc_vs_threshold", "ROC vs threshold"),
            ("plot_f1_accuracy_vs_threshold", "F1/Accuracy vs threshold"),
            ("plot_calibration_curve", "Calibration curve"),
            ("plot_gradcam_comparison", "Grad-CAM comparison"),
        ]
        for func_name, desc in functions:
            if not check_function_exists(plotting, func_name, desc):
                all_passed = False
    except ImportError:
        all_passed = False
    
    # 4. Check UI features
    print("\n4. Checking UI Features:")
    print("-" * 60)
    ui_file = Path("ui/app.py")
    if ui_file.exists():
        with open(ui_file, 'r') as f:
            ui_content = f.read()
        
        features = [
            ("Batch Upload", "batch" in ui_content.lower() and "upload" in ui_content.lower()),
            ("Grad-CAM Methods", "gradcam" in ui_content.lower() or "grad_cam" in ui_content.lower()),
            ("Uncertainty", "uncertainty" in ui_content.lower() or "monte_carlo" in ui_content.lower()),
            ("Model Transparency", "transparency" in ui_content.lower() or "model info" in ui_content.lower()),
        ]
        
        for feature_name, found in features:
            if found:
                print(f"✓ {feature_name}: Found in UI")
            else:
                print(f"✗ {feature_name}: NOT FOUND in UI")
                all_passed = False
    else:
        print("✗ UI file not found: ui/app.py")
        all_passed = False
    
    # 5. Check required output files structure
    print("\n5. Checking Output Directory Structure:")
    print("-" * 60)
    results_dir = Path("results")
    if results_dir.exists():
        print(f"✓ Results directory exists: {results_dir}")
    else:
        print(f"✗ Results directory missing: {results_dir}")
        all_passed = False
    
    # 6. Check documentation
    print("\n6. Checking Documentation:")
    print("-" * 60)
    docs_to_check = [
        ("README.md", "Main README"),
        ("IMPROVEMENTS.md", "Improvements documentation"),
        ("REVIEW_IMPROVEMENTS_SUMMARY.md", "Review improvements summary"),
    ]
    
    for doc_file, desc in docs_to_check:
        if not check_file_exists(doc_file, desc):
            all_passed = False
    
    # 7. Check docs/figs directory
    print("\n7. Checking IEEE Report Figures Directory:")
    print("-" * 60)
    docs_figs = Path("docs/figs")
    if docs_figs.exists():
        print(f"✓ docs/figs directory exists")
    else:
        print(f"⚠ docs/figs directory missing (will be created by prepare_ieee_figures.py)")
    
    # 8. Check ablation study exports
    print("\n8. Checking Ablation Study Export Format:")
    print("-" * 60)
    ablation_file = Path("src/ablation_study.py")
    if ablation_file.exists():
        with open(ablation_file, 'r') as f:
            ablation_content = f.read()
        
        has_csv = "csv" in ablation_content.lower() and "to_csv" in ablation_content
        has_json = "json" in ablation_content.lower() and "json.dump" in ablation_content
        
        if has_csv:
            print("✓ CSV export: Found")
        else:
            print("✗ CSV export: NOT FOUND")
            all_passed = False
        
        if has_json:
            print("✓ JSON export: Found")
        else:
            print("✗ JSON export: NOT FOUND")
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
        print("\nDeliverable 3 requirements appear to be met!")
    else:
        print("✗ SOME CHECKS FAILED")
        print("\nPlease fix the issues above before submitting.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


