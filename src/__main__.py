#!/usr/bin/env python3
"""
Entry point for src module.

Usage: python -m src.train --epochs 5
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main entry point for src module."""
    parser = argparse.ArgumentParser(description="Medical X-ray Triage src module")
    parser.add_argument('subcommand', choices=['train', 'eval', 'interpret', 'make_sample_data'],
                       help='Subcommand to run')
    
    if len(sys.argv) < 2:
        parser.print_help()
        return
    
    subcommand = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'src' from argv
    
    if subcommand == 'train':
        from .train import main as train_main
        train_main()
    elif subcommand == 'eval':
        from .eval import main as eval_main
        eval_main()
    elif subcommand == 'interpret':
        from .interpret import main as interpret_main
        interpret_main()
    elif subcommand == 'make_sample_data':
        from .make_sample_data import main as make_sample_main
        make_sample_main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

