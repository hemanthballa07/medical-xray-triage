"""
Comprehensive integrity test for the Medical X-ray Triage System.

This script tests all modules, imports, and functionality to ensure
the system works correctly after implementing all review suggestions.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("1. TESTING MODULE IMPORTS")
    print("=" * 60)
    
    modules_to_test = [
        ("config", "src.config"),
        ("data", "src.data"),
        ("model", "src.model"),
        ("train", "src.train"),
        ("eval", "src.eval"),
        ("eval_enhanced", "src.eval_enhanced"),
        ("interpret", "src.interpret"),
        ("utils", "src.utils"),
        ("uncertainty", "src.uncertainty"),
        ("plotting", "src.plotting"),
        ("bootstrap_metrics", "src.bootstrap_metrics"),
        ("failure_analysis", "src.failure_analysis"),
        ("cross_dataset_eval", "src.cross_dataset_eval"),
        ("hyperparameter_sweep", "src.hyperparameter_sweep"),
        ("audit_module", "src.audit_module"),
        ("ablation_study", "src.ablation_study"),
        ("generate_additional_plots", "src.generate_additional_plots"),
    ]
    
    results = []
    for name, module_path in modules_to_test:
        try:
            __import__(module_path)
            print(f"✅ {name}")
            results.append((name, True, None))
        except Exception as e:
            print(f"❌ {name}: {e}")
            results.append((name, False, str(e)))
    
    return results


def test_function_signatures():
    """Test critical function signatures."""
    print("\n" + "=" * 60)
    print("2. TESTING FUNCTION SIGNATURES")
    print("=" * 60)
    
    import inspect
    
    results = []
    
    # Test train_epoch
    try:
        from train import train_epoch
        sig = inspect.signature(train_epoch)
        params = list(sig.parameters.keys())
        expected = ['model', 'train_loader', 'criterion', 'optimizer', 'device']
        if params == expected:
            print(f"✅ train_epoch: {params}")
            results.append(("train_epoch", True))
        else:
            print(f"❌ train_epoch: Expected {expected}, got {params}")
            results.append(("train_epoch", False))
    except Exception as e:
        print(f"❌ train_epoch: {e}")
        results.append(("train_epoch", False))
    
    # Test validate_epoch
    try:
        from train import validate_epoch
        sig = inspect.signature(validate_epoch)
        params = list(sig.parameters.keys())
        expected = ['model', 'val_loader', 'criterion', 'device']
        if params == expected:
            print(f"✅ validate_epoch: {params}")
            results.append(("validate_epoch", True))
        else:
            print(f"❌ validate_epoch: Expected {expected}, got {params}")
            results.append(("validate_epoch", False))
    except Exception as e:
        print(f"❌ validate_epoch: {e}")
        results.append(("validate_epoch", False))
    
    return results


def test_hyperparameter_sweep():
    """Test hyperparameter sweep module."""
    print("\n" + "=" * 60)
    print("3. TESTING HYPERPARAMETER SWEEP")
    print("=" * 60)
    
    try:
        from hyperparameter_sweep import run_hyperparameter_sweep, objective
        import inspect
        
        # Check objective function signature
        sig = inspect.signature(objective)
        params = list(sig.parameters.keys())
        if 'trial' in params and 'data_dir' in params:
            print(f"✅ objective function signature: {params}")
        else:
            print(f"⚠️  objective function signature: {params}")
        
        # Check if Optuna is available
        try:
            import optuna
            print("✅ Optuna is available")
        except ImportError:
            print("⚠️  Optuna not installed (optional)")
        
        return True
    except Exception as e:
        print(f"❌ Hyperparameter sweep: {e}")
        return False


def test_audit_module():
    """Test audit module."""
    print("\n" + "=" * 60)
    print("4. TESTING AUDIT MODULE")
    print("=" * 60)
    
    try:
        from audit_module import (
            compute_subgroup_metrics,
            analyze_fairness,
            visualize_subgroup_metrics,
            audit_model_performance
        )
        print("✅ All audit module functions available")
        return True
    except Exception as e:
        print(f"❌ Audit module: {e}")
        return False


def test_failure_analysis():
    """Test failure analysis module."""
    print("\n" + "=" * 60)
    print("5. TESTING FAILURE ANALYSIS")
    print("=" * 60)
    
    try:
        from failure_analysis import (
            identify_failure_cases,
            visualize_failure_cases,
            analyze_and_visualize_failures
        )
        print("✅ All failure analysis functions available")
        return True
    except Exception as e:
        print(f"❌ Failure analysis: {e}")
        return False


def test_bootstrap_metrics():
    """Test bootstrap metrics module."""
    print("\n" + "=" * 60)
    print("6. TESTING BOOTSTRAP METRICS")
    print("=" * 60)
    
    try:
        from bootstrap_metrics import (
            bootstrap_metric,
            bootstrap_all_metrics,
            plot_bootstrap_distributions
        )
        print("✅ All bootstrap metrics functions available")
        return True
    except Exception as e:
        print(f"❌ Bootstrap metrics: {e}")
        return False


def test_ui_imports():
    """Test UI can import required modules."""
    print("\n" + "=" * 60)
    print("7. TESTING UI IMPORTS")
    print("=" * 60)
    
    ui_path = Path(__file__).parent / "ui" / "app.py"
    if not ui_path.exists():
        print("❌ UI file not found")
        return False
    
    # Check if UI can import required modules
    try:
        sys.path.append(str(Path(__file__).parent))
        # Just check imports, don't run Streamlit
        with open(ui_path, 'r') as f:
            content = f.read()
        
        required_imports = [
            'streamlit',
            'torch',
            'model',
            'data',
            'utils',
            'plotting',
            'uncertainty'
        ]
        
        missing = []
        for imp in required_imports:
            if imp not in content.lower():
                missing.append(imp)
        
        if missing:
            print(f"⚠️  Potentially missing imports: {missing}")
        else:
            print("✅ UI file structure looks good")
        
        return True
    except Exception as e:
        print(f"❌ UI check: {e}")
        return False


def test_docker_files():
    """Test Docker configuration files."""
    print("\n" + "=" * 60)
    print("8. TESTING DOCKER CONFIGURATION")
    print("=" * 60)
    
    dockerfile = Path(__file__).parent / "Dockerfile"
    docker_compose = Path(__file__).parent / "docker-compose.yml"
    dockerignore = Path(__file__).parent / ".dockerignore"
    
    results = []
    
    if dockerfile.exists():
        print("✅ Dockerfile exists")
        results.append(True)
    else:
        print("❌ Dockerfile missing")
        results.append(False)
    
    if docker_compose.exists():
        print("✅ docker-compose.yml exists")
        results.append(True)
    else:
        print("❌ docker-compose.yml missing")
        results.append(False)
    
    if dockerignore.exists():
        print("✅ .dockerignore exists")
        results.append(True)
    else:
        print("⚠️  .dockerignore missing (optional)")
        results.append(True)  # Not critical
    
    return all(results)


def test_requirements():
    """Test requirements.txt completeness."""
    print("\n" + "=" * 60)
    print("9. TESTING REQUIREMENTS.TXT")
    print("=" * 60)
    
    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    with open(req_file, 'r') as f:
        requirements = f.read()
    
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'scikit-learn',
        'pillow',
        'matplotlib',
        'streamlit',
        'tqdm',
        'pyyaml',
        'pytorch-grad-cam',
        'opencv-python',
        'psutil',
        'optuna',
        'scipy',
        'seaborn',
        'plotly',
        'kaleido'
    ]
    
    missing = []
    for pkg in required_packages:
        if pkg.lower() not in requirements.lower():
            missing.append(pkg)
    
    if missing:
        print(f"⚠️  Potentially missing packages: {missing}")
        return False
    else:
        print("✅ All required packages in requirements.txt")
        return True


def test_file_structure():
    """Test that all expected files exist."""
    print("\n" + "=" * 60)
    print("10. TESTING FILE STRUCTURE")
    print("=" * 60)
    
    required_files = [
        "src/train.py",
        "src/eval.py",
        "src/eval_enhanced.py",
        "src/interpret.py",
        "src/model.py",
        "src/data.py",
        "src/utils.py",
        "src/config.py",
        "src/uncertainty.py",
        "src/plotting.py",
        "src/bootstrap_metrics.py",
        "src/failure_analysis.py",
        "src/cross_dataset_eval.py",
        "src/hyperparameter_sweep.py",
        "src/audit_module.py",
        "src/ablation_study.py",
        "ui/app.py",
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "README.md"
    ]
    
    missing = []
    for file_path in required_files:
        full_path = Path(__file__).parent / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} MISSING")
            missing.append(file_path)
    
    return len(missing) == 0


def main():
    """Run all integrity tests."""
    print("\n" + "=" * 60)
    print("MEDICAL X-RAY TRIAGE SYSTEM - INTEGRITY CHECK")
    print("=" * 60)
    print()
    
    results = {}
    
    # Run all tests
    results['imports'] = test_imports()
    results['signatures'] = test_function_signatures()
    results['hyperparameter'] = test_hyperparameter_sweep()
    results['audit'] = test_audit_module()
    results['failure'] = test_failure_analysis()
    results['bootstrap'] = test_bootstrap_metrics()
    results['ui'] = test_ui_imports()
    results['docker'] = test_docker_files()
    results['requirements'] = test_requirements()
    results['structure'] = test_file_structure()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        if isinstance(result, list):
            # Handle both 2-tuple and 3-tuple results
            passed = sum(1 for item in result if (isinstance(item, tuple) and len(item) >= 2 and item[1]) or (isinstance(item, bool) and item))
            total = len(result)
            total_tests += total
            passed_tests += passed
            print(f"{test_name}: {passed}/{total} passed")
        elif isinstance(result, bool):
            total_tests += 1
            if result:
                passed_tests += 1
                print(f"{test_name}: ✅ PASSED")
            else:
                print(f"{test_name}: ❌ FAILED")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All integrity checks passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Please review above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

