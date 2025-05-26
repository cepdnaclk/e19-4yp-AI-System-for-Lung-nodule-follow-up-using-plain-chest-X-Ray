from ct_xray_core import CTtoXrayConverter

# CLI Interface for the converter
class CLI:
    def __init__(self):
        self.converter = CTtoXrayConverter()
    
    def run(self, args):
        """Run the CLI with the given arguments"""
        import argparse
        
        parser = argparse.ArgumentParser(description='Convert CT images to X-ray projections')
        
        # Input options
        input_group = parser.add_argument_group('Input options')
        input_group.add_argument('-i', '--input', required=True, help='Input CT image file or DICOM directory')
        input_group.add_argument('-s', '--series', action='store_true', help='Input is a DICOM series directory')
        
        # Output options
        output_group = parser.add_argument_group('Output options')
        output_group.add_argument('-o', '--output', required=True, help='Output X-ray image file')
        
        # Processing options
        process_group = parser.add_argument_group('Processing options')
        process_group.add_argument('-a', '--axis', type=int, choices=[0, 1, 2], default=1,
                                  help='Projection axis (0=X, 1=Y, 2=Z)')
        process_group.add_argument('--min', type=float, default=0.0, help='Minimum attenuation coefficient')
        process_group.add_argument('--max', type=float, default=0.001, help='Maximum attenuation coefficient')
        process_group.add_argument('--clip', type=float, default=1.0, help='Percentile for contrast clipping')
        process_group.add_argument('--blur', type=float, default=0.5, help='Sigma for Gaussian blur (0 for no blur)')
        process_group.add_argument('--no-invert', action='store_true', help='Do not invert the X-ray image')
        
        parsed_args = parser.parse_args(args)
        
        # Load the CT image
        if parsed_args.series:
            success, message = self.converter.load_ct_image_from_series(parsed_args.input)
        else:
            success, message = self.converter.load_ct_image_from_file(parsed_args.input)
        
        if not success:
            print(f"Error: {message}")
            return 1
        
        print(message)
        
        # Set parameters
        self.converter.set_parameter('projection_axis', parsed_args.axis)
        self.converter.set_parameter('mu_min', parsed_args.min)
        self.converter.set_parameter('mu_max', parsed_args.max)
        self.converter.set_parameter('clip_percentile', parsed_args.clip)
        self.converter.set_parameter('blur_sigma', parsed_args.blur)
        self.converter.set_parameter('invert_xray', not parsed_args.no_invert)
        
        # Convert to X-ray
        print("Converting to X-ray...")
        success, message = self.converter.convert_to_xray()
        
        if not success:
            print(f"Error: {message}")
            return 1
        
        print(message)
        
        # Save the X-ray image
        print(f"Saving X-ray to {parsed_args.output}...")
        success, message = self.converter.save_xray_image(parsed_args.output)
        
        if not success:
            print(f"Error: {message}")
            return 1
        
        print(message)
        return 0
    