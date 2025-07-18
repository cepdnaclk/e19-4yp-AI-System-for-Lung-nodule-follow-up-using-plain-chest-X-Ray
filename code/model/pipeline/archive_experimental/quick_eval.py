"""
Quick evaluation script for testing the improved pipeline.
"""

import torch
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from evaluate import ModelEvaluator
from code.model.pipeline.config_ import Config

def quick_evaluate(model_path):
    """Quick evaluation function."""
    print(f"Evaluating model: {model_path}")
    
    # Create config
    config = Config()
    config.create_directories()
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, config)
    
    # Run evaluation
    metrics_default, metrics_optimal, predictions = evaluator.evaluate_model(save_results=True)
    
    print("Evaluation completed!")
    return metrics_default, metrics_optimal

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Evaluation')
    parser.add_argument('--model_path', type=str, 
                       default='./checkpoints/best_model.pth',
                       help='Path to the trained model checkpoint')
    
    args = parser.parse_args()
    
    try:
        quick_evaluate(args.model_path)
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
