import os
import sys
import argparse
from ct_xray_core import CTtoXrayConverter
import traceback

def find_ct_folders(input_root_dir, min_dicom_files=10):
    """
    Find all folders that likely contain CT DICOM files by recursively traversing directories
    
    Args:
        input_root_dir: Root directory to start search
        min_dicom_files: Minimum number of DICOM files required (to avoid single-file folders)
        
    Returns:
        List of tuples (folder_path, relative_path)
    """
    ct_folders = []
    
    # Helper function for recursive traversal
    def traverse(current_dir, relative_dir=""):
        # Check if this folder contains DICOM files
        files = os.listdir(current_dir)
        dicom_files = [f for f in files if f.lower().endswith(('.dcm', '.ima'))]
        
        # If folder has enough DICOM files, it's likely a CT folder
        if len(dicom_files) >= min_dicom_files:
            # Use original relative_dir if it exists, otherwise use folder name
            rel_path = relative_dir if relative_dir else os.path.basename(current_dir)
            ct_folders.append((current_dir, rel_path))
            # Don't traverse deeper if we found DICOM files here
            return
        
        # Otherwise check subfolders
        subdirs = [d for d in files if os.path.isdir(os.path.join(current_dir, d))]
        for subdir in subdirs:
            subdir_path = os.path.join(current_dir, subdir)
            new_relative = os.path.join(relative_dir, subdir) if relative_dir else subdir
            traverse(subdir_path, new_relative)
    
    # Start traversal from root
    traverse(input_root_dir)
    return ct_folders


def process_ct_folder(input_folder, output_folder, output_prefix="", axis=1, params=None):
    """
    Process a single CT folder to generate X-ray and mask
    
    Args:
        input_folder: Path to folder containing CT DICOM series
        output_folder: Path to save the X-ray and mask
        output_prefix: Prefix to use for output filenames
        axis: Projection axis (0=X, 1=Y, 2=Z)
        params: Optional dictionary of parameters for the converter
    
    Returns:
        Tuple of (success, message)
    """
    # Create converter instance
    converter = CTtoXrayConverter()
    
    # Set parameters if provided
    if params:
        for key, value in params.items():
            converter.set_parameter(key, value)
    
    # Set projection axis
    converter.set_parameter('projection_axis', axis)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the CT series
    print(f"Loading CT series from {input_folder}...")
    success, message = converter.load_ct_image_from_series(input_folder)
    if not success:
        return False, f"Failed to load CT series: {message}"
    
    print(f"Loaded CT series with shape {converter.ct_image.shape}")
    
    # Convert to X-ray
    print("Converting to X-ray...")
    success, message = converter.convert_to_xray()
    if not success:
        return False, f"Failed to convert to X-ray: {message}"
    
    # Save the X-ray image and mask
    output_path = os.path.join(output_folder, f"{output_prefix}_drr.png")
    print(f"Saving X-ray to {output_path}...")
    
    # The save_xray_image method already adds _mask to the filename for the mask
    # Just need to modify how it constructs the mask filename in ct_xray_core.py
    success, message = converter.save_xray_image(output_path)
    
    return success, message

def batch_process(input_root_dir, output_root_dir, axis=1, params=None):
    """
    Process all CT folders in the input directory, searching recursively
    
    Args:
        input_root_dir: Root directory containing CT folders (possibly nested)
        output_root_dir: Root directory to save output files (single folder)
        axis: Projection axis (0=X, 1=Y, 2=Z)
        params: Optional dictionary of parameters for the converter
    
    Returns:
        Number of successfully processed folders
    """
    # Create output root directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)
    
    # Find all CT folders by recursively searching the directory structure
    ct_folders = find_ct_folders(input_root_dir)
    
    if not ct_folders:
        print(f"No CT folders found in {input_root_dir}")
        return 0
    
    print(f"Found {len(ct_folders)} CT folders to process")
    
    # Process each folder
    success_count = 0
    for i, (folder_path, relative_path) in enumerate(ct_folders, 1):
        # Create output prefix by replacing path separators with underscores
        output_prefix = relative_path.replace('\\', '_').replace('/', '_')
        
        print(f"\n[{i}/{len(ct_folders)}] Processing folder: {relative_path}")
        try:
            success, message = process_ct_folder(folder_path, output_root_dir, output_prefix, axis, params)
            if success:
                print(f"Success: {message}")
                success_count += 1
            else:
                print(f"Error: {message}")
        except Exception as e:
            print(f"Exception while processing {relative_path}: {str(e)}")
            traceback.print_exc()
    
    return success_count
    
def main():
    parser = argparse.ArgumentParser(description='Batch convert CT folders to X-ray projections')
    
    parser.add_argument('-i', '--input', required=True, 
                        help='Input root directory containing CT dataset folders')
    parser.add_argument('-o', '--output', required=True,
                        help='Output root directory for saving X-rays and masks')
    parser.add_argument('-a', '--axis', type=int, choices=[0, 1, 2], default=1,
                       help='Projection axis (0=X, 1=Y, 2=Z)')
    
    # Add additional parameters
    parser.add_argument('--min', type=float, default=0.0, 
                       help='Minimum attenuation coefficient')
    parser.add_argument('--max', type=float, default=0.001,
                       help='Maximum attenuation coefficient')
    parser.add_argument('--clip', type=float, default=1.0,
                       help='Percentile for contrast clipping')
    parser.add_argument('--blur', type=float, default=0.5,
                       help='Sigma for Gaussian blur (0 for no blur)')
    parser.add_argument('--no-invert', action='store_true',
                       help='Do not invert the X-ray image')
    
    args = parser.parse_args()
    
    # Prepare parameters
    params = {
        'mu_min': args.min,
        'mu_max': args.max,
        'clip_percentile': args.clip,
        'blur_sigma': args.blur,
        'invert_xray': not args.no_invert,
    }
    
    # Process folders
    print(f"Starting batch processing...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    
    success_count = batch_process(args.input, args.output, args.axis, params)
    
    print(f"\nCompleted processing {success_count} folders successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())