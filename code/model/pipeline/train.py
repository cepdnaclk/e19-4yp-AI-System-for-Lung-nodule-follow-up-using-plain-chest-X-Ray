"""
Simple training script for testing the improved pipeline.
"""

import torch
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import all necessary modules
from main import main

if __name__ == "__main__":
    print("Starting improved training pipeline...")
    try:
        main()
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
