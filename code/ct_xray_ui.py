import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import threading
import numpy as np
import tkinter as tk

# UI Class for Tkinter Application
class CTtoXrayUI:
    def __init__(self, root, converter):
        
        self.root = root
        self.root.title("CT to X-ray Converter")
        self.root.geometry("1000x700")
        
        # Create the converter instance (business logic)
        self.converter = converter
        
        # Variables
        self.slice_var = tk.IntVar(value=0)  # For slice slider
        self.projection_axis = tk.IntVar(value=1)  # Default: y-axis (1)
        self.mu_min = tk.DoubleVar(value=0.0)
        self.mu_max = tk.DoubleVar(value=0.001)
        self.clip_percentile = tk.DoubleVar(value=1.0)
        self.blur_sigma = tk.DoubleVar(value=0.5)
        self.invert_xray = tk.BooleanVar(value=True)
        
        # Setup UI
        self.setup_ui()
        
        # Link variables to the converter
        self.link_variables()
    
    def link_variables(self):
        """Link Tkinter variables to the converter parameters"""
        self.projection_axis.trace_add("write", lambda *args: self.converter.set_parameter('projection_axis', self.projection_axis.get()))
        self.mu_min.trace_add("write", lambda *args: self.converter.set_parameter('mu_min', self.mu_min.get()))
        self.mu_max.trace_add("write", lambda *args: self.converter.set_parameter('mu_max', self.mu_max.get()))
        self.clip_percentile.trace_add("write", lambda *args: self.converter.set_parameter('clip_percentile', self.clip_percentile.get()))
        self.blur_sigma.trace_add("write", lambda *args: self.converter.set_parameter('blur_sigma', self.blur_sigma.get()))
        self.invert_xray.trace_add("write", lambda *args: self.converter.set_parameter('invert_xray', self.invert_xray.get()))
    
    def setup_ui(self):
        """Setup the UI components"""
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        
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
        load_button = tk.Button(load_frame, text="Load Single File", command=self.load_single_file)
        load_button.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        load_series_button = tk.Button(load_frame, text="Load DICOM Series", command=self.load_dicom_series)
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
    
    def load_single_file(self):
        """Load a single CT file"""
        import tkinter.filedialog as filedialog
        
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
        
        success, message = self.converter.load_ct_image_from_file(filepath)
        
        if success:
            self.status_var.set(message)
            
            # Setup display based on image type
            if len(self.converter.ct_image.shape) == 3:
                self.setup_slice_slider()
                slice_idx = self.converter.ct_image.shape[0] // 2
                self.slice_var.set(slice_idx)
                self.display_ct_slice(slice_idx)
            else:
                self.slice_frame.pack_forget()  # Hide slice slider
                self.display_ct_image()
        else:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", message)
            self.status_var.set("Error loading image")
    
    def load_dicom_series(self):
        """Load a DICOM series from a directory"""
        import tkinter.filedialog as filedialog
        
        directory = filedialog.askdirectory(
            title="Select Directory with DICOM Series"
        )
        
        if not directory:
            return
            
        self.status_var.set(f"Loading DICOM series from {os.path.basename(directory)}...")
        self.root.update_idletasks()
        
        success, message = self.converter.load_ct_image_from_series(directory)
        
        if success:
            self.status_var.set(message)
            
            # Setup slice slider for 3D volume
            self.setup_slice_slider()
            
            # Display middle slice
            slice_idx = self.converter.ct_image.shape[0] // 2
            self.slice_var.set(slice_idx)
            self.display_ct_slice(slice_idx)
        else:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Error", message)
            self.status_var.set("Error loading DICOM series")
    
    def setup_slice_slider(self):
        """Configure the slice slider based on the loaded volume"""
        num_slices = self.converter.get_slice_count()
        
        if num_slices > 0:
            # Show the slice frame
            self.slice_frame.pack(side=tk.LEFT, padx=20, fill=tk.Y)
            
            # Update slider configuration
            self.slice_slider.configure(from_=0, to=num_slices-1)
            self.slice_label.configure(text=f"Slice: 0/{num_slices-1}")
    
    def update_slice_view(self, event=None):
        """Update the displayed CT slice based on slider position"""
        if self.converter.ct_image is None:
            return
            
        slice_idx = self.slice_var.get()
        num_slices = self.converter.get_slice_count()
        
        if num_slices > 0:
            # Update the slice label
            self.slice_label.configure(text=f"Slice: {slice_idx}/{num_slices-1}")
            
            # Display the selected slice
            self.display_ct_slice(slice_idx)
    
    def display_ct_slice(self, slice_idx):
        """Display a single slice from the CT volume"""
        self.ax1.clear()
        
        display_data = self.converter.get_normalized_slice(slice_idx)
        if display_data is not None:
            self.ax1.imshow(display_data, cmap='gray')
            self.ax1.set_title(f"CT Slice {slice_idx}")
            self.ax1.axis('off')
            
            # Overlay annotations for nodules (if available for this slice)
            if hasattr(self.converter, "z_to_slice_map") and hasattr(self.converter, "nodule_annotations"):
                for z, idx in self.converter.z_to_slice_map.items():
                    if idx == slice_idx:
                        for coords in self.converter.nodule_annotations[z]:
                            coords_array = np.array(coords)
                            if np.any(np.isnan(coords_array)) or len(coords_array) < 3:
                                continue  # Skip bad or incomplete polygons
                            
                            # Step 1: Convert to (x, y)
                            coords_array = coords_array[:, ::-1]

                            # Step 2: Sort polygon points by angle around centroid
                            centroid = np.mean(coords_array, axis=0)
                            angles = np.arctan2(coords_array[:,1] - centroid[1], coords_array[:,0] - centroid[0])
                            sorted_indices = np.argsort(angles)
                            sorted_coords = coords_array[sorted_indices]

                            # Draw the polygon
                            polygon = patches.Polygon(
                                sorted_coords,
                                closed=True,
                                edgecolor='red',
                                facecolor='none',
                                linewidth=1.5
                            )
                            self.ax1.add_patch(polygon)
                            
                            # coords_array = coords_array[:, ::-1]  # Swap back to (x, y) for CT display
                            # polygon = patches.Polygon(
                            #     coords_array,
                            #     closed=True,
                            #     edgecolor='red',
                            #     facecolor='none',  # Transparent fill
                            #     linewidth=1.5
                            # )
                            # self.ax1.add_patch(polygon)
            
            self.canvas.draw()
    
    def display_ct_image(self):
        """Display the CT image"""
        self.ax1.clear()
        
        display_data = self.converter.get_normalized_ct_image()
        if display_data is not None:
            self.ax1.imshow(display_data, cmap='gray')
            self.ax1.set_title("CT Image")
            self.ax1.axis('off')
            self.canvas.draw()
    
    def display_xray(self):
        """Display the generated X-ray image"""
        self.ax2.clear()

        if self.converter.xray_image is not None:
            # Flip the X-ray image vertically
            flipped_xray = np.flipud(self.converter.xray_image)

            # Calculate aspect ratio based on voxel spacing
            aspect_ratio = 1  # Default
            if self.converter.spacing is not None:
                aspect_ratio = self.converter.spacing[2] / self.converter.spacing[1]  # y-spacing / z-spacing

            self.ax2.imshow(flipped_xray, cmap='gray', aspect=aspect_ratio)
             
            # Overlay projected nodule mask
            if self.converter.projected_nodule_mask is not None:
                flipped_mask = np.flipud(self.converter.projected_nodule_mask)

                # Overlay with same aspect ratio as DRR
                contour = self.ax2.contour(
                    flipped_mask, 
                    levels=[0.5], 
                    colors='red', 
                    linewidths=1.5
                )
            self.ax2.set_title("X-ray Projection")
            self.ax2.axis('off')
            self.fig.tight_layout()
            self.canvas.draw()
    
    def convert_thread(self):
        """Run conversion in a separate thread to keep UI responsive"""
        import tkinter.messagebox as messagebox
        
        if self.converter.ct_image is None:
            messagebox.showwarning("Warning", "Please load a CT image first")
            return
        
        self.status_var.set("Converting to X-ray...")
        self.root.update_idletasks()
        
        # Create thread for conversion
        thread = threading.Thread(target=self.run_conversion)
        thread.daemon = True
        thread.start()
    
    def run_conversion(self):
        """Run the conversion process in a thread"""
        success, message = self.converter.convert_to_xray()
        
        # Update UI from the main thread
        self.root.after(0, lambda: self.status_var.set(message))
        
        if success:
            self.root.after(0, self.display_xray)
    
    def save_xray_image(self):
        """Save the X-ray image to a file"""
        import tkinter.filedialog as filedialog
        import tkinter.messagebox as messagebox
        
        if self.converter.xray_image is None:
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
        
        success, message = self.converter.save_xray_image(filepath)
        
        if success:
            self.status_var.set(message)
        else:
            messagebox.showerror("Error", message)
            self.status_var.set("Error saving image")
    
    def show_advanced_settings(self):
        """Show window with advanced X-ray conversion settings"""
        import tkinter as tk

        
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

