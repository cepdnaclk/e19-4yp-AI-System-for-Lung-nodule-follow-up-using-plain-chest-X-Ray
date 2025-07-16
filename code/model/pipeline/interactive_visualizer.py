"""
Interactive visualization script for exploring specific samples.
Easy-to-use interface for visualizing model internals.
"""

import os
import sys
sys.path.append('.')

from visualize_model_internals import visualize_specific_sample, ModelInternalVisualizer
from small_dataset_config import SmallDatasetConfig
from drr_dataset_loading import DRRDataset


def show_dataset_info():
    """Show information about the dataset."""
    config = SmallDatasetConfig()
    
    # Load both datasets to get info
    try:
        train_dataset = DRRDataset(
            data_root=config.DATA_ROOT,
            image_size=config.IMAGE_SIZE,
            training=True,
            augment=False,
            normalize=config.NORMALIZE_DATA
        )
        
        val_dataset = DRRDataset(
            data_root=config.DATA_ROOT,
            image_size=config.IMAGE_SIZE,
            training=False,
            augment=False,
            normalize=config.NORMALIZE_DATA
        )
        
        print(f"Dataset Information:")
        print(f"  - Training samples: {len(train_dataset)} (indices: 0 to {len(train_dataset)-1})")
        print(f"  - Validation samples: {len(val_dataset)} (indices: 0 to {len(val_dataset)-1})")
        print(f"  - Total samples: {len(train_dataset) + len(val_dataset)}")
        print(f"  - Image size: {config.IMAGE_SIZE}")
        print(f"  - Data root: {config.DATA_ROOT}")
        
        return len(train_dataset), len(val_dataset)
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 0, 0


def interactive_mode():
    """Interactive mode for sample selection."""
    print("="*60)
    print("  Nodule-Specific Model Visualization Tool")
    print("="*60)
    
    # Show dataset info
    train_samples, val_samples = show_dataset_info()
    if train_samples == 0 and val_samples == 0:
        print("Cannot load dataset. Exiting.")
        return
    
    print(f"\nThis tool will generate comprehensive visualizations showing:")
    print(f"  • Input X-ray and DRR images")
    print(f"  • Ground truth mask overlay")
    print(f"  • Spatial attention maps")
    print(f"  • Nodule-specific feature maps")
    print(f"  • Fused feature representations")
    print(f"  • Final segmentation predictions")
    print(f"  • Feature map channels for different layers")
    print(f"  • Attention analysis with statistics")
    
    while True:
        print(f"\n" + "-"*40)
        print("Options:")
        print("  1. Visualize specific sample")
        print("  2. Visualize multiple samples") 
        print("  3. Compare samples side-by-side")
        print("  4. Show dataset info")
        print("  5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Single sample visualization
            try:
                print("\nChoose dataset:")
                print("  1. Training set")
                print("  2. Validation set")
                dataset_choice = input("Enter choice (1-2): ").strip()
                
                if dataset_choice == '1':
                    from_training = True
                    max_samples = train_samples
                    dataset_name = "training"
                elif dataset_choice == '2':
                    from_training = False
                    max_samples = val_samples
                    dataset_name = "validation"
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    continue
                
                sample_idx = int(input(f"Enter sample index from {dataset_name} set (0-{max_samples-1}): "))
                if 0 <= sample_idx < max_samples:
                    output_dir = input("Output directory (default: 'sample_visualizations'): ").strip()
                    if not output_dir:
                        output_dir = 'sample_visualizations'
                    
                    print(f"\nGenerating visualizations for sample {sample_idx} from {dataset_name} set...")
                    visualize_specific_sample(sample_idx, output_dir, from_training)
                    
                else:
                    print(f"Invalid sample index. Must be between 0 and {max_samples-1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            # Multiple samples
            try:
                print("\nChoose dataset:")
                print("  1. Training set")
                print("  2. Validation set")
                dataset_choice = input("Enter choice (1-2): ").strip()
                
                if dataset_choice == '1':
                    from_training = True
                    max_samples = train_samples
                    dataset_name = "training"
                elif dataset_choice == '2':
                    from_training = False
                    max_samples = val_samples
                    dataset_name = "validation"
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    continue
                
                samples_input = input(f"Enter sample indices from {dataset_name} set separated by commas (e.g., 0,5,10): ").strip()
                sample_indices = [int(x.strip()) for x in samples_input.split(',')]
                
                # Validate indices
                invalid_indices = [i for i in sample_indices if i < 0 or i >= max_samples]
                if invalid_indices:
                    print(f"Invalid sample indices: {invalid_indices}. Must be between 0 and {max_samples-1}")
                    continue
                
                output_dir = input("Output directory (default: 'multi_sample_visualizations'): ").strip()
                if not output_dir:
                    output_dir = 'multi_sample_visualizations'
                
                print(f"\nGenerating visualizations for samples {sample_indices} from {dataset_name} set...")
                for idx in sample_indices:
                    print(f"Processing sample {idx}...")
                    visualize_specific_sample(idx, output_dir, from_training)
                
                print(f"✓ All visualizations saved to {output_dir}/")
                
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            # Side-by-side comparison
            try:
                print("\nChoose dataset:")
                print("  1. Training set")
                print("  2. Validation set")
                dataset_choice = input("Enter choice (1-2): ").strip()
                
                if dataset_choice == '1':
                    from_training = True
                    max_samples = train_samples
                    dataset_name = "training"
                elif dataset_choice == '2':
                    from_training = False
                    max_samples = val_samples
                    dataset_name = "validation"
                else:
                    print("Invalid choice. Please enter 1 or 2.")
                    continue
                
                num_samples = int(input("Number of samples to compare (default: 5): ") or "5")
                start_idx = int(input(f"Starting sample index from {dataset_name} set (default: 0): ") or "0")
                
                if start_idx + num_samples > max_samples:
                    print(f"Not enough samples. Maximum starting index: {max_samples - num_samples}")
                    continue
                
                output_dir = input("Output directory (default: 'comparison_visualizations'): ").strip()
                if not output_dir:
                    output_dir = 'comparison_visualizations'
                
                print(f"\nCreating comparison of samples {start_idx} to {start_idx + num_samples - 1} from {dataset_name} set...")
                
                # Create visualizer and comparison
                config = SmallDatasetConfig()
                model_paths = [
                    os.path.join(config.SAVE_DIR, 'best_lightweight_model.pth'),
                    os.path.join(config.SAVE_DIR, 'final_lightweight_model.pth')
                ]
                
                model_path = None
                for path in model_paths:
                    if os.path.exists(path):
                        model_path = path
                        break
                
                visualizer = ModelInternalVisualizer(model_path, config)
                os.makedirs(output_dir, exist_ok=True)
                
                # Create custom comparison for specific range from specified dataset
                visualizer.compare_multiple_samples_range_dataset(
                    start_idx=start_idx, 
                    num_samples=num_samples, 
                    save_dir=output_dir,
                    from_training=from_training
                )
                
                print(f"✓ Comparison visualization saved to {output_dir}/")
                
            except ValueError:
                print("Invalid input. Please enter numbers.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            # Show dataset info again
            show_dataset_info()
        
        elif choice == '5':
            print("Exiting visualization tool. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-5.")


def quick_visualize(sample_idx):
    """Quick visualization function for scripting."""
    try:
        sample_idx = int(sample_idx)
        print(f"Quick visualization of sample {sample_idx}...")
        visualize_specific_sample(sample_idx, f'quick_vis_sample_{sample_idx}')
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line mode - quick visualization
        try:
            sample_idx = int(sys.argv[1])
            quick_visualize(sample_idx)
        except (ValueError, IndexError):
            print("Usage: python interactive_visualizer.py [sample_index]")
            print("   or: python interactive_visualizer.py (for interactive mode)")
    else:
        # Interactive mode
        interactive_mode()
