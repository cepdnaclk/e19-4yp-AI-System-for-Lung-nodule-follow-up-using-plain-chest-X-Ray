import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, exposure, transform
import SimpleITK as sitk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import glob
from scipy.ndimage import gaussian_filter

class CTtoXrayConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("CT to X-ray Converter")
        self.root.geometry("1000x700")
        
        # Variables
        self.ct_image = None
        self.xray_image = None
        self.projection_axis = tk.IntVar(value=1)  # Default: y-axis (1)
        self.slice_var = tk.IntVar(value=0)  # For slice slider
        
        # Advanced settings
        self.mu_min = tk.DoubleVar(value=0.0)  # Minimum attenuation coefficient
        self.mu_max = tk.DoubleVar(value=0.001)  # Maximum attenuation coefficient
        self.clip_percentile = tk.DoubleVar(value=1.0)  # Percentile for contrast clipping
        self.blur_sigma = tk.DoubleVar(value=0.5)  # Sigma for Gaussian blur
        self.invert_xray = tk.BooleanVar(value=True)  # Whether to invert the X-ray
        
        # Setup UI
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        # Load buttons frame
        load_frame = tk.Frame(control_frame)
        load_frame.pack(side=tk.LEFT, padx=5)
        
        # Load buttons
        load_button = tk.Button(load_frame, text="Load Single File", command=lambda: self.load_ct_image(mode="single"))
        load_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        load_series_button = tk.Button(load_frame, text="Load DICOM Series", command=lambda: self.load_ct_image(mode="series"))
        load_series_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Process buttons
        process_frame = tk.Frame(control_frame)
        process_frame.pack(side=tk.LEFT, padx=5)
        
        # Convert button
        convert_button = tk.Button(process_frame, text="Convert to X-ray", command=self.convert_thread)
        convert_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Save button
        save_button = tk.Button(process_frame, text="Save X-ray Image", command=self.save_xray_image)
        save_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        # Projection axis selection
        axis_frame = tk.LabelFrame(control_frame, text="Projection Axis")
        axis_frame.pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(axis_frame, text="X-axis", variable=self.projection_axis, value=0).pack(anchor=tk.W)
        tk.Radiobutton(axis_frame, text="Y-axis", variable=self.projection_axis, value=1).pack(anchor=tk.W)
        tk.Radiobutton(axis_frame, text="Z-axis", variable=self.projection_axis, value=2).pack(anchor=tk.W)
        
        # Advanced settings button
        settings_button = tk.Button(control_frame, text="Advanced Settings", command=self.show_advanced_settings)
        settings_button.pack(side=tk.LEFT, padx=10)
        
        # Slice selector (for 3D volumes)
        self.slice_frame = tk.LabelFrame(control_frame, text="CT Slice")
        self.slice_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        self.slice_var = tk.IntVar(value=0)
        self.slice_label = tk.Label(self.slice_frame, text="Slice: 0/0")
        self.slice_label.pack(anchor=tk.W, pady=2)
        
        self.slice_slider = ttk.Scale(
            self.slice_frame, 
            from_=0, 
            to=100,
            orient=tk.HORIZONTAL,
            length=150,
            variable=self.slice_var,
            command=self.update_slice_view
        )
        self.slice_slider.pack(anchor=tk.W, pady=2)
        self.slice_frame.pack_forget()  # Initially hidden
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(control_frame, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        
        # Image display area
        self.display_frame = tk.Frame(main_frame)
        self.display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create figure for matplotlib
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Initial plot setup
        self.ax1.set_title("CT Image")
        self.ax2.set_title("X-ray Projection")
        self.ax1.axis('off')
        self.ax2.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def show_advanced_settings(self):
        """Show window with advanced X-ray conversion settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Advanced X-ray Settings")
        settings_window.geometry("350x250")
        settings_window.transient(self.root)  # Make window modal
        settings_window.grab_set()  # Make window modal
        
        # Create settings frames
        frame = tk.Frame(settings_window, padx=10, pady=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Attenuation coefficient range
        tk.Label(frame, text="Minimum attenuation coefficient:").grid(row=0, column=0, sticky="w", pady=2)
        tk.Entry(frame, textvariable=self.mu_min, width=10).grid(row=0, column=1, sticky="w", pady=2)
        
        tk.Label(frame, text="Maximum attenuation coefficient:").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(frame, textvariable=self.mu_max, width=10).grid(row=1, column=1, sticky="w", pady=2)
        
        # Contrast clipping
        tk.Label(frame, text="Contrast clip percentile:").grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(frame, textvariable=self.clip_percentile, width=10).grid(row=2, column=1, sticky="w", pady=2)
        
        # Blur amount
        tk.Label(frame, text="Blur sigma (0 for no blur):").grid(row=3, column=0, sticky="w", pady=2)
        tk.Entry(frame, textvariable=self.blur_sigma, width=10).grid(row=3, column=1, sticky="w", pady=2)
        
        # Invert X-ray option
        tk.Checkbutton(frame, text="Invert X-ray (like traditional film)", 
                      variable=self.invert_xray).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        # Reset to defaults button
        def reset_defaults():
            self.mu_min.set(0.0)
            self.mu_max.set(0.5)
            self.clip_percentile.set(1.0)
            self.blur_sigma.set(0.5)
            self.invert_xray.set(True)
            
        reset_button = tk.Button(frame, text="Reset to Defaults", command=reset_defaults)
        reset_button.grid(row=5, column=0, pady=10, sticky="w")
        
        # Close button
        close_button = tk.Button(frame, text="Close", command=settings_window.destroy)
        close_button.grid(row=5, column=1, pady=10, sticky="e")
    
    def load_ct_image(self, mode="single"):
        """Load a CT image - either a single file or a directory of DICOM files"""
        if mode == "series":
            # Load a DICOM series from a directory
            directory = filedialog.askdirectory(
                title="Select Directory with DICOM Series"
            )
            
            if not directory:
                return
                
            self.status_var.set(f"Loading DICOM series from {os.path.basename(directory)}...")
            self.root.update_idletasks()
            
            try:
                # Use SimpleITK to read the DICOM series
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(directory)
                
                if not dicom_names:
                    # Try direct file search if GDCM didn't find any series
                    dicom_names = sorted(glob.glob(os.path.join(directory, "*.dcm")))
                    
                    if not dicom_names:
                        messagebox.showerror("Error", "No DICOM files found in the selected directory")
                        self.status_var.set("No DICOM files found")
                        return
                
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                self.ct_image = sitk.GetArrayFromImage(image)
                
                # Setup slice slider for 3D volume
                self.setup_slice_slider()
                
                # Display middle slice
                slice_idx = self.ct_image.shape[0] // 2
                self.slice_var.set(slice_idx)
                self.display_ct_slice(slice_idx)
                self.status_var.set(f"Loaded DICOM series with {len(dicom_names)} files, shape {self.ct_image.shape}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load DICOM series: {str(e)}")
                self.status_var.set("Error loading DICOM series")
        
        else:  # mode == "single"
            # Load a single file (standard option)
            filetypes = [
                ("DICOM files", "*.dcm"),
                ("NIFTI files", "*.nii *.nii.gz"),
                ("TIFF stacks", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
            
            filepath = filedialog.askopenfilename(
                title="Select CT Image File",
                filetypes=filetypes
            )
            
            if not filepath:
                return
            
            self.status_var.set(f"Loading {os.path.basename(filepath)}...")
            self.root.update_idletasks()
            
            try:
                # Attempt to read with SimpleITK (handles DICOM, NIFTI, etc.)
                image = sitk.ReadImage(filepath)
                self.ct_image = sitk.GetArrayFromImage(image)
                
                # If it's a 3D volume
                if len(self.ct_image.shape) == 3:
                    # Setup slice slider for 3D volume
                    self.setup_slice_slider()
                    
                    # Display middle slice
                    slice_idx = self.ct_image.shape[0] // 2
                    self.slice_var.set(slice_idx)
                    self.display_ct_slice(slice_idx)
                    self.status_var.set(f"Loaded CT volume with shape {self.ct_image.shape}")
                else:
                    # It's a 2D image
                    self.slice_frame.pack_forget()  # Hide slice slider
                    self.display_ct_image()
                    self.status_var.set(f"Loaded 2D image with shape {self.ct_image.shape}")
            
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
                            
                        self.slice_frame.pack_forget()  # Hide slice slider
                        self.display_ct_image()
                    elif len(self.ct_image.shape) == 3:
                        # This is a volume (stack of 2D images)
                        self.setup_slice_slider()
                        slice_idx = self.ct_image.shape[0] // 2
                        self.slice_var.set(slice_idx)
                        self.display_ct_slice(slice_idx)
                    else:
                        # It's a 2D image
                        self.slice_frame.pack_forget()  # Hide slice slider
                        self.display_ct_image()
                        
                    self.status_var.set(f"Loaded image with shape {self.ct_image.shape}")
                    
                except Exception as e2:
                    messagebox.showerror("Error", f"Failed to load image: {str(e2)}")
                    self.status_var.set("Error loading image")
                    
    def setup_slice_slider(self):
        """Configure the slice slider based on the loaded volume"""
        if len(self.ct_image.shape) == 3:
            num_slices = self.ct_image.shape[0]
            
            # Show the slice frame
            self.slice_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
            
            # Update slider configuration
            self.slice_slider.configure(from_=0, to=num_slices-1)
            self.slice_label.configure(text=f"Slice: 0/{num_slices-1}")
            
    def update_slice_view(self, event=None):
        """Update the displayed CT slice based on slider position"""
        if self.ct_image is None or len(self.ct_image.shape) < 3:
            return
            
        slice_idx = self.slice_var.get()
        num_slices = self.ct_image.shape[0]
        
        # Update the slice label
        self.slice_label.configure(text=f"Slice: {slice_idx}/{num_slices-1}")
        
        # Display the selected slice
        self.display_ct_slice(slice_idx)
    
    def display_ct_slice(self, slice_idx):
        """Display a single slice from the CT volume"""
        self.ax1.clear()
        
        # Normalize for display if needed
        slice_data = self.ct_image[slice_idx]
        if np.min(slice_data) != np.max(slice_data):  # Avoid division by zero
            p_low, p_high = np.percentile(slice_data, [1, 99])
            display_data = exposure.rescale_intensity(slice_data, in_range=(p_low, p_high))
        else:
            display_data = slice_data
            
        self.ax1.imshow(display_data, cmap='gray')
        self.ax1.set_title(f"CT Slice {slice_idx}")
        self.ax1.axis('off')
        self.canvas.draw()
    
    def display_ct_image(self):
        """Display the CT image"""
        self.ax1.clear()
        
        # Normalize for display if needed
        if np.min(self.ct_image) != np.max(self.ct_image):  # Avoid division by zero
            p_low, p_high = np.percentile(self.ct_image, [1, 99])
            display_data = exposure.rescale_intensity(self.ct_image, in_range=(p_low, p_high))
        else:
            display_data = self.ct_image
            
        self.ax1.imshow(display_data, cmap='gray')
        self.ax1.set_title("CT Image")
        self.ax1.axis('off')
        self.canvas.draw()
    
    def convert_thread(self):
        """Run conversion in a separate thread to keep UI responsive"""
        if self.ct_image is None:
            messagebox.showwarning("Warning", "Please load a CT image first")
            return
        
        self.status_var.set("Converting to X-ray...")
        self.root.update_idletasks()
        
        # Create thread for conversion
        thread = threading.Thread(target=self.convert_to_xray)
        thread.daemon = True
        thread.start()
    
    def convert_to_xray(self):
        """Convert the CT volume to an X-ray-like projection using Beer-Lambert law"""
        try:
            # Ensure we have a 3D volume
            if len(self.ct_image.shape) < 3:
                # For a 2D image, we'll just process it and treat it as an X-ray already
                self.xray_image = self.process_xray_image(self.ct_image)
                self.status_var.set("Processed 2D image as X-ray")
                self.display_xray()
                return
            
            # Get the projection axis
            axis = self.projection_axis.get()
            
            # Get the advanced parameters
            mu_min = self.mu_min.get()
            mu_max = self.mu_max.get()
            clip_percentile = self.clip_percentile.get()
            blur_sigma = self.blur_sigma.get()
            
            # Step 1: Scale the CT values to attenuation coefficients
            hu_min, hu_max = np.min(self.ct_image), np.max(self.ct_image)
            mu_volume = mu_min + (self.ct_image - hu_min) * (mu_max - mu_min) / (hu_max - hu_min)
            
            # Step 2: Apply Beer-Lambert law: I = I_0 * exp(-∫μ(x)dx)
            attenuation_sum = np.sum(mu_volume, axis=axis)
            initial_intensity = 1.0
            xray = initial_intensity * np.exp(-attenuation_sum)
            
            # Process the raw X-ray image for better visualization
            self.xray_image = self.process_xray_image(xray, clip_percentile, blur_sigma)
            
            # Display the result
            self.display_xray()
            self.status_var.set("X-ray conversion complete")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error during conversion: {str(e)}")
            self.status_var.set("Conversion failed")
    
    def process_xray_image(self, xray_raw, clip_percentile=None, blur_sigma=None):
        """Process the raw X-ray image for better visualization"""
        if clip_percentile is None:
            clip_percentile = self.clip_percentile.get()
        if blur_sigma is None:
            blur_sigma = self.blur_sigma.get()         
            
        # Clip outliers for better contrast
        p_low, p_high = np.percentile(xray_raw, [clip_percentile, 100-clip_percentile])
        xray = exposure.rescale_intensity(xray_raw, in_range=(p_low, p_high), out_range=(0, 1))
        
        # Apply adaptive histogram equalization for better detail visibility
        xray = exposure.equalize_adapthist(xray, clip_limit=0.03)
        
        # Apply mild Gaussian blur to simulate scatter
        if blur_sigma > 0:
            xray = gaussian_filter(xray, sigma=blur_sigma)
        
        # Invert if traditional X-ray look is desired (white bones on black background)
        if self.invert_xray.get():
            xray = 1 - xray
            
        return xray
    
    def display_xray(self):
        """Display the generated X-ray image"""
        self.ax2.clear()

        # Flip the X-ray image vertically
        flipped_xray = np.flipud(self.xray_image)

        self.ax2.imshow(flipped_xray, cmap='gray' , aspect='auto')
        self.ax2.set_title("X-ray Projection")
        self.ax2.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def save_xray_image(self):
        """Save the X-ray image to a file"""
        if self.xray_image is None:
            messagebox.showwarning("Warning", "No X-ray image to save")
            return
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("TIFF files", "*.tif"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save X-ray Image",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if not filepath:
            return
        
        try:
            # Convert to 8-bit for common image formats
            xray_8bit = (self.xray_image * 255).astype(np.uint8)
            io.imsave(filepath, xray_8bit)
            self.status_var.set(f"Saved to {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")
            self.status_var.set("Error saving image")


def main():
    root = tk.Tk()
    app = CTtoXrayConverter(root)
    root.mainloop()


if __name__ == "__main__":
    main()