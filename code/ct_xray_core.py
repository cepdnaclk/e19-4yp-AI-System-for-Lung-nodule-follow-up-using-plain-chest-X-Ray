import os
import glob
import numpy as np
import SimpleITK as sitk
from skimage import io, exposure
from scipy.ndimage import gaussian_filter
import xml.etree.ElementTree as ET
from skimage.draw import polygon2mask


class CTtoXrayConverter:
    """
    Business logic for converting CT images to X-ray projections.
    """
    def __init__(self):
        # Default parameters
        self.ct_image_sitk = None;
        self.ct_image = None
        self.xray_image = None
        self.spacing = None
        self.projected_nodule_mask = None
        self.params = {
            'projection_axis': 1,  # Default: y-axis (1)
            'mu_min': 0.0,  # Minimum attenuation coefficient
            'mu_max': 0.001,  # Maximum attenuation coefficient
            'clip_percentile': 1.0,  # Percentile for contrast clipping
            'blur_sigma': 0.5,  # Sigma for Gaussian blur
            'invert_xray': True,  # Whether to invert the X-ray
        }
        
    def load_ct_image_from_file(self, filepath):
        """Load a CT image from a single file"""
        try:
            # Attempt to read with SimpleITK (handles DICOM, NIFTI, etc.)
            image = sitk.ReadImage(filepath)
            self.ct_image_sitk = image
            self.ct_image = sitk.GetArrayFromImage(image)
            
            # Get spacing information if available
            self.spacing = image.GetSpacing()
            
            return True, f"Loaded image with shape {self.ct_image.shape}"
            
        except Exception as e:
            # Fallback to skimage for other formats
            try:
                self.ct_image = io.imread(filepath)
                
                # Handle 3D stack from some image formats
                if len(self.ct_image.shape) == 3 and self.ct_image.shape[2] in [1, 3, 4]:
                    # This is probably a color image, not a volume
                    # Convert to grayscale if needed
                    if self.ct_image.shape[2] == 3:  # RGB
                        self.ct_image = np.mean(self.ct_image, axis=2).astype(self.ct_image.dtype)
                    elif self.ct_image.shape[2] == 4:  # RGBA
                        self.ct_image = np.mean(self.ct_image[:,:,0:3], axis=2).astype(self.ct_image.dtype)
                    else:  # Single channel
                        self.ct_image = self.ct_image[:,:,0]
                
                return True, f"Loaded image with shape {self.ct_image.shape}"
                
            except Exception as e2:
                return False, f"Failed to load image: {str(e2)}"
    
    def load_ct_image_from_series(self, directory):
        """Load a CT image from a DICOM series directory"""
        try:
            # Use SimpleITK to read the DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(directory)
            
            if not dicom_names:
                # Try direct file search if GDCM didn't find any series
                dicom_names = sorted(glob.glob(os.path.join(directory, "*.dcm")))
                
                if not dicom_names:
                    return False, "No DICOM files found in the selected directory"
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            self.ct_image_sitk = image
            self.ct_image = sitk.GetArrayFromImage(image)

            # Extract voxel spacing
            self.spacing = image.GetSpacing()

            # After loading CT images, load the annotation file from the same directory
            self.load_annotation_from_ct_dir(directory)
            
            return True, f"Loaded DICOM series with {len(dicom_names)} files, shape {self.ct_image.shape}"
            
        except Exception as e:
            return False, f"Failed to load DICOM series: {str(e)}"
    
    def load_nodule_annotations(self, xml_path):
        """Load nodule annotations from the XML file."""
        self.nodule_annotations = {}  # Dictionary: {z_position: [list of (x, y) coords]}

        tree = ET.parse(xml_path)
        root = tree.getroot()

        ns = {'ns': 'http://www.nih.gov'}  # Namespace for XML

        for nodule in root.findall('.//ns:unblindedReadNodule', ns):
            for roi in nodule.findall('ns:roi', ns):
                included = roi.find('ns:inclusion', ns)
                if included is not None and included.text.strip().upper() == "TRUE":
                    z = float(roi.find('ns:imageZposition', ns).text)
                    # Round z to 1 decimal place to match slice map keys
                    z = round(z, 1)
                    coords = []
                    for edge in roi.findall('ns:edgeMap', ns):
                        x = int(edge.find('ns:xCoord', ns).text)
                        y = int(edge.find('ns:yCoord', ns).text)
                        coords.append((y, x))
                    if z not in self.nodule_annotations:
                        self.nodule_annotations[z] = []
                    self.nodule_annotations[z].append(coords)
        
        # Debug check
        print(f"Parsed {len(self.nodule_annotations)} z-slices with annotations")
        for z, polygons in self.nodule_annotations.items():
            print(f"Z={z} has {len(polygons)} polygons")

    def load_annotation_from_ct_dir(self, ct_directory):
        xml_file = None
        for f in os.listdir(ct_directory):
            if f.lower().endswith('.xml'):
                xml_file = os.path.join(ct_directory, f)
                break

        if xml_file:
            self.load_nodule_annotations(xml_file)
            print(f"XML file loaded")
            print("CT image type:", type(self.ct_image))
            self.map_z_positions_to_slices()
            print(f"Mapping is successful!")

        else:
            print("No XML annotation file found in the directory.")

    def map_z_positions_to_slices(self):
        image = self.ct_image_sitk
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()
        size = image.GetSize()

        # Z-axis location for each slice
        slice_locations = [
            origin[2] + i * spacing[2] * direction[8]  # Z-axis index
            for i in range(size[2])
        ]

        self.z_to_slice_map = {}
        for z in self.nodule_annotations:
            closest = min(range(len(slice_locations)), key=lambda i: abs(slice_locations[i] - z))
            self.z_to_slice_map[z] = closest
        
        # Debugging steps
        print(self.z_to_slice_map)
        print(f"Spacing: {spacing}")
        print(f"Origin: {origin}")
        print(f"Direction: {direction}")
    
    def get_slice(self, slice_idx):
        """Get a specific slice from the CT volume"""
        if self.ct_image is None or len(self.ct_image.shape) < 3:
            return None
        
        return self.ct_image[slice_idx]
    
    def get_normalized_slice(self, slice_idx):
        """Get a normalized slice for display"""
        slice_data = self.get_slice(slice_idx)
        if slice_data is None:
            return None
            
        # Normalize for display
        if np.min(slice_data) != np.max(slice_data):  # Avoid division by zero
            p_low, p_high = np.percentile(slice_data, [1, 99])
            return exposure.rescale_intensity(slice_data, in_range=(p_low, p_high))
        else:
            return slice_data
    
    def get_normalized_ct_image(self):
        """Get the normalized CT image for display"""
        if self.ct_image is None:
            return None
            
        # Normalize for display
        if np.min(self.ct_image) != np.max(self.ct_image):  # Avoid division by zero
            p_low, p_high = np.percentile(self.ct_image, [1, 99])
            return exposure.rescale_intensity(self.ct_image, in_range=(p_low, p_high))
        else:
            return self.ct_image
    
    def set_parameter(self, param_name, value):
        """Set a parameter for the conversion"""
        if param_name in self.params:
            self.params[param_name] = value
            return True
        return False
    
    def get_parameter(self, param_name):
        """Get a parameter value"""
        return self.params.get(param_name)
    
    def get_slice_count(self):
        """Get the number of slices in the CT volume"""
        if self.ct_image is None or len(self.ct_image.shape) < 3:
            return 0
        return self.ct_image.shape[0]
    
    def process_xray_image(self, xray_raw, clip_percentile=None, blur_sigma=None, invert=None):
        """Process the raw X-ray image for better visualization"""
        if clip_percentile is None:
            clip_percentile = self.params['clip_percentile']
        if blur_sigma is None:
            blur_sigma = self.params['blur_sigma']
        if invert is None:
            invert = self.params['invert_xray']
            
        # Clip outliers for better contrast
        p_low, p_high = np.percentile(xray_raw, [clip_percentile, 100-clip_percentile])
        xray = exposure.rescale_intensity(xray_raw, in_range=(p_low, p_high), out_range=(0, 1))
        
        # Apply adaptive histogram equalization for better detail visibility
        xray = exposure.equalize_adapthist(xray, clip_limit=0.03)
        
        # Apply mild Gaussian blur to simulate scatter
        if blur_sigma > 0:
            xray = gaussian_filter(xray, sigma=blur_sigma)
        
        # Invert if traditional X-ray look is desired (white bones on black background)
        if invert:
            xray = 1 - xray
            
        return xray

    def generate_nodule_mask_volume(self):
        """Generate a 3D binary mask from parsed XML nodule annotations"""
        if self.ct_image is None or not hasattr(self, 'nodule_annotations'):
            return None

        shape = self.ct_image.shape  # (Z, Y, X)
        mask_volume = np.zeros(shape, dtype=np.uint8)

        for z_value, polygons in self.nodule_annotations.items():
            slice_idx = self.z_to_slice_map.get(z_value)
            if slice_idx is None or slice_idx >= shape[0]:
                continue

            for coords in polygons:
                if len(coords) < 3:
                    continue
                # mask = polygon2mask(
                #     shape=(shape[1], shape[2]),  # (Y, X)
                #     polygon=np.array(coords)
                # )
                mask = polygon2mask((shape[1], shape[2]), np.array(coords))
                mask_volume[slice_idx] |= mask.astype(np.uint8)

        return mask_volume
    
    def convert_to_xray(self):
        """Convert the CT volume to an X-ray-like projection using Beer-Lambert law"""
        if self.ct_image is None:
            return False, "No CT image loaded"
            
        try:
            # Ensure we have a 3D volume
            if len(self.ct_image.shape) < 3:
                # For a 2D image, we'll just process it and treat it as an X-ray already
                self.xray_image = self.process_xray_image(self.ct_image)
                return True, "Processed 2D image as X-ray"
            
            # Get the projection axis
            axis = self.params['projection_axis']
            
            # Get the parameters
            mu_min = self.params['mu_min']
            mu_max = self.params['mu_max']
            
            # Remove air/background (e.g., threshold below -500 HU)
            body_mask = self.ct_image > -500  # adjust threshold as needed
            ct_body_only = np.where(body_mask, self.ct_image, -1000)  # replace background with air (very low HU)
            
            # Step 1: Scale the CT values to attenuation coefficients
            hu_min, hu_max = np.min(ct_body_only), np.max(ct_body_only)
            mu_volume = mu_min + (ct_body_only - hu_min) * (mu_max - mu_min) / (hu_max - hu_min)
            
            # Step 2: Apply Beer-Lambert law: I = I_0 * exp(-∫μ(x)dx)
            attenuation_sum = np.sum(mu_volume, axis=axis)
            initial_intensity = 1.0
            xray = initial_intensity * np.exp(-attenuation_sum)
            
            # Process the raw X-ray image for better visualization
            self.xray_image = self.process_xray_image(xray)

            # Generate projected annotation mask
            mask_volume = self.generate_nodule_mask_volume()
            if mask_volume is not None:
                self.projected_nodule_mask = np.max(mask_volume, axis=axis)
            else:
                self.projected_nodule_mask = None
            
            return True, "X-ray conversion complete"
        
        except Exception as e:
            return False, f"Error during conversion: {str(e)}"
    
    def save_xray_image(self, filepath):
        """Save the X-ray image to a file"""
        if self.xray_image is None:
            return False, "No X-ray image to save"
        
        try:
            # Convert to 8-bit for common image formats
            xray_8bit = (self.xray_image * 255).astype(np.uint8)
            io.imsave(filepath, xray_8bit)
            return True, f"Saved to {os.path.basename(filepath)}"
        except Exception as e:
            return False, f"Failed to save image: {str(e)}"
