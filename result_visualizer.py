import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import os

# Define base paths as specified
PRE_IMAGES_BASE_DIR = r"C:\\Users\\Cynix\\Desktop\\Sri charan working on this folder\\CGNet_workingOnBackBone_ssd-done_workingOnChangingAttenionModule\\dataset"
# POST_IMAGES_BASE_DIR is the same as PRE_IMAGES_BASE_DIR in this case
CHANGE_MAPS_RESULTS_DIR = r"C:\\Users\\Cynix\\Desktop\\Sri charan working on this folder\\CGNet_workingOnBackBone_ssd-done_workingOnChangingAttenionModule\\test_result"

class ImageTripletVisualizer:

    def __init__(self, master):
        self.master = master
        self.master.title("Change Detection Result Visualizer")

        self.test_set_paths = [] # List of relative paths to test set folders
        self.current_test_set_index = -1
        self.image_files_in_current_test_set = [] # List of image filenames (e.g., test_1.png)
        self.current_image_file_index = -1
        self.image_display_size = (300, 300)
        self.original_image_pil = {"pre": None, "post": None, "change": None}

        # --- Magnifier Attributes ---
        self.magnifier_window = None
        self.magnifier_label = None
        self.magnifier_size = 150  # The size of the square magnifier window
        self.magnification_factor = 3  # How much to zoom in

        # --- UI Elements ---
        top_frame = ttk.Frame(master, padding="10")
        top_frame.pack(fill=tk.X)

        # Result Type Selection (ASPP/Non-ASPP/All)
        ttk.Label(top_frame, text="Result Type:").grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        self.result_type_combobox = ttk.Combobox(top_frame, state="readonly", width=15)
        self.result_type_combobox['values'] = ["All", "ASPP", "Non-ASPP"]
        self.result_type_combobox.set("All")
        self.result_type_combobox.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.result_type_combobox.bind("<<ComboboxSelected>>", self.on_result_type_select)

        # Test Set Selection
        ttk.Label(top_frame, text="Test Set:").grid(row=0, column=2, sticky=tk.W, padx=(10,5))
        self.test_set_combobox = ttk.Combobox(top_frame, state="readonly", width=45)
        self.test_set_combobox.grid(row=0, column=3, sticky=tk.EW, padx=5)
        self.test_set_combobox.bind("<<ComboboxSelected>>", self.on_test_set_select)

        scan_button = ttk.Button(top_frame, text="Scan/Refresh", command=self.scan_and_load_test_sets)
        scan_button.grid(row=0, column=4, padx=5)

        # Image File Selection within a Test Set
        ttk.Label(top_frame, text="Image File:").grid(row=1, column=0, sticky=tk.W, padx=(0,5), pady=(5,0))
        self.image_file_combobox = ttk.Combobox(top_frame, state="readonly", width=60)
        self.image_file_combobox.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=(5,0), columnspan=2)
        self.image_file_combobox.bind("<<ComboboxSelected>>", self.on_image_file_select)

        self.prev_button = ttk.Button(top_frame, text="Previous Image", command=self.show_previous_image)
        self.prev_button.grid(row=1, column=3, padx=5, pady=(5,0))

        self.next_button = ttk.Button(top_frame, text="Next Image", command=self.show_next_image)
        self.next_button.grid(row=1, column=4, padx=5, pady=(5,0))

        top_frame.columnconfigure(3, weight=1)


        # Frame for images
        image_frame = ttk.Frame(master, padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)

        self.image_pil_references = {} 
        self.image_display_widgets = {}
        titles_and_keys = [("Pre Image", "pre"), ("Post Image", "post"), ("Change Map", "change")]

        for i, (title_text, key) in enumerate(titles_and_keys):
            img_sub_frame = ttk.Frame(image_frame)
            img_sub_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            ttk.Label(img_sub_frame, text=title_text, font=("Arial", 12, "bold")).pack()
            
            img_widget = ttk.Label(img_sub_frame, background="lightgrey")
            img_widget.pack(fill=tk.BOTH, expand=True)
            self.image_display_widgets[key] = img_widget
            
            # --- Bind events for the new magnifier ---
            img_widget.bind("<Enter>", self._create_magnifier)
            img_widget.bind("<Leave>", self._destroy_magnifier)
            img_widget.bind("<Motion>", self._update_magnifier)


        self.scan_and_load_test_sets()

    def on_result_type_select(self, event=None):
        self.scan_and_load_test_sets()

    def _find_test_sets(self, scan_path):
        found_test_sets = []
        if not os.path.isdir(scan_path):
            print(f"Error: Change maps directory '{scan_path}' does not exist.")
            return found_test_sets

        # Determine filter based on result type
        result_type = self.result_type_combobox.get() if hasattr(self, 'result_type_combobox') else "All"

        # Always collect all test sets, but filter at the dataset (first-level) granularity
        dataset_dirs = [d for d in os.listdir(scan_path) if os.path.isdir(os.path.join(scan_path, d))]
        for dataset_dir in sorted(dataset_dirs):
            dataset_path = os.path.join(scan_path, dataset_dir)
            # For each dataset, look for subfolders (model/experiment folders)
            for root, dirs, files in os.walk(dataset_path):
                if any(f.lower().endswith(".png") for f in files):
                    relative_path = os.path.relpath(root, scan_path)
                    if relative_path and relative_path != ".":
                        rel_lower = relative_path.lower()
                        # Filtering logic: only filter on the experiment/model folder, not the dataset folder
                        # e.g., relative_path = 'LEVIR-CD-256/CGNet-resnet34_ASPP' -> check only after first '/'
                        parts = relative_path.split(os.sep)
                        exp_folder = parts[1] if len(parts) > 1 else ""
                        if result_type == "ASPP" and "aspp" not in exp_folder.lower():
                            continue
                        if result_type == "Non-ASPP" and "aspp" in exp_folder.lower():
                            continue
                        found_test_sets.append(relative_path)
        found_test_sets.sort()
        return found_test_sets

    def scan_and_load_test_sets(self):
        self.test_set_paths = self._find_test_sets(CHANGE_MAPS_RESULTS_DIR)
        
        if not self.test_set_paths:
            self.test_set_combobox['values'] = ["No test sets found"]
            self.test_set_combobox.set("No test sets found")
            self.current_test_set_index = -1
            self._clear_all_image_displays()
            self._clear_image_file_list()
            self.update_navigation_buttons_state()
            print(f"No subdirectories with PNGs found in {CHANGE_MAPS_RESULTS_DIR}")
            return

        self.test_set_combobox['values'] = self.test_set_paths
        self.current_test_set_index = 0
        self.test_set_combobox.set(self.test_set_paths[self.current_test_set_index])
        self.load_images_for_selected_test_set()

    def on_test_set_select(self, event=None):
        selected_test_set_rel_path = self.test_set_combobox.get()
        try:
            self.current_test_set_index = self.test_set_paths.index(selected_test_set_rel_path)
            self.load_images_for_selected_test_set()
        except ValueError:
            print(f"Error: Could not find selected test set \'{selected_test_set_rel_path}\'")
            self._clear_all_image_displays()
            self._clear_image_file_list()

    def load_images_for_selected_test_set(self):
        if not (0 <= self.current_test_set_index < len(self.test_set_paths)):
            self._clear_all_image_displays()
            self._clear_image_file_list()
            return

        current_test_set_rel_path = self.test_set_paths[self.current_test_set_index]
        full_test_set_path = os.path.join(CHANGE_MAPS_RESULTS_DIR, current_test_set_rel_path)

        self.image_files_in_current_test_set = sorted([
            f for f in os.listdir(full_test_set_path) 
            if os.path.isfile(os.path.join(full_test_set_path, f)) and f.lower().endswith(".png")
        ])

        if not self.image_files_in_current_test_set:
            self._clear_all_image_displays()
            self._clear_image_file_list("No PNGs in this set")
            print(f"No PNG image files found in {full_test_set_path}")
            return
        
        self.image_file_combobox['values'] = self.image_files_in_current_test_set
        self.current_image_file_index = 0
        self.image_file_combobox.set(self.image_files_in_current_test_set[self.current_image_file_index])
        self.load_current_image_triplet()
        self.update_navigation_buttons_state()

    def on_image_file_select(self, event=None):
        selected_image_file = self.image_file_combobox.get()
        try:
            self.current_image_file_index = self.image_files_in_current_test_set.index(selected_image_file)
            self.load_current_image_triplet()
        except ValueError:
            print(f"Error: Could not find selected image file \'{selected_image_file}\'")
            self._clear_all_image_displays()

    def load_current_image_triplet(self):
        if not (0 <= self.current_test_set_index < len(self.test_set_paths)) or \
           not (0 <= self.current_image_file_index < len(self.image_files_in_current_test_set)):
            self._clear_all_image_displays()
            return
        
        test_set_rel_path = self.test_set_paths[self.current_test_set_index]
        current_change_map_filename = self.image_files_in_current_test_set[self.current_image_file_index]

        # Extract dataset_name from the test_set_rel_path
        path_parts = test_set_rel_path.split(os.sep)
        if not path_parts:
            print(f"Error: Could not determine dataset name from path {test_set_rel_path}")
            self._clear_all_image_displays()
            return
        dataset_name = path_parts[0]

        # Map dataset names to their actual directory names if they differ
        dataset_dir_mapping = {
            "SYSU": "SYSU-CD",  # Add mapping for SYSU to SYSU-CD
            # Add other mappings here if needed
        }
        
        # Use the mapped directory name if it exists, otherwise use the original
        dataset_dir = dataset_dir_mapping.get(dataset_name, dataset_name)

        # Attempt to strip known model-related prefixes to get the original base filename for pre/post images
        original_filename_for_dataset = current_change_map_filename
        prefix_stripped = False  # Initialize the flag here

        # Special handling for SYSU-CD dataset
        if dataset_name == "SYSU-CD" or dataset_name == "SYSU":
            # Handle filenames like CGNet03996.png -> 03996.png
            if current_change_map_filename.startswith("CGNet"):
                try:
                    # Extract the number part (e.g., 03996) from CGNet03996.png
                    num_part = current_change_map_filename[5:]  # Remove 'CGNet'
                    original_filename_for_dataset = num_part
                    print(f"INFO: Converted SYSU-CD filename from '{current_change_map_filename}' to '{original_filename_for_dataset}'")
                    prefix_stripped = True  # Set flag to True after successful conversion
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not convert SYSU-CD filename '{current_change_map_filename}': {e}")
                    return
        else:
            # Original prefix stripping logic for other datasets
            known_prefixes_in_filenames = ["CGNet", "CGN", "HCGM"]
            temp_filename_lower = current_change_map_filename.lower()
            prefix_stripped = False
            for pf in known_prefixes_in_filenames:
                if temp_filename_lower.startswith(pf.lower()):
                    remainder = current_change_map_filename[len(pf):]
                    if remainder and (remainder.lower().startswith("test") or remainder.lower().startswith("val")):
                        original_filename_for_dataset = remainder
                        print(f"INFO: Stripped prefix '{pf}' from '{current_change_map_filename}', using '{original_filename_for_dataset}' for pre/post images.")
                        prefix_stripped = True
                        break

        if not prefix_stripped and dataset_name not in ["SYSU-CD", "SYSU"]:  # Skip this message for SYSU dataset
            print(f"INFO: No known prefix stripped from '{current_change_map_filename}'. Using it as is for pre/post images.")

        # For SYSU-CD dataset, check if the file exists before proceeding
        if dataset_name == "SYSU-CD" or dataset_name == "SYSU":
            pre_path = os.path.join(PRE_IMAGES_BASE_DIR, dataset_dir, "test", "A", original_filename_for_dataset)
            if not os.path.exists(pre_path):
                print(f"Warning: Pre-image not found at {pre_path}")
                self._clear_all_image_displays()
                return

        paths_to_load = {
            "change": os.path.join(CHANGE_MAPS_RESULTS_DIR, test_set_rel_path, current_change_map_filename),
            "pre": os.path.join(PRE_IMAGES_BASE_DIR, dataset_dir, "test", "A", original_filename_for_dataset),
            "post": os.path.join(PRE_IMAGES_BASE_DIR, dataset_dir, "test", "B", original_filename_for_dataset)
        }

        for key, img_path in paths_to_load.items():
            try:
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"File not found: {img_path}")
                
                img = Image.open(img_path)
                # Store the original full-resolution image
                self.original_image_pil[key] = img.copy() # Store a copy
                # Initial display will be handled by _update_displayed_images via _reset_zoom or load_images_for_selected_test_set

            except FileNotFoundError:
                print(f"Warning: Image not found at {img_path}")
                self.original_image_pil[key] = None # Ensure it's None if not found
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                self.original_image_pil[key] = None # Ensure it's None on error
        
        # Now that originals are loaded (or confirmed None), update the display based on current zoom state
        self._update_displayed_images()
        self.update_navigation_buttons_state()
        if self.image_files_in_current_test_set:
            self.image_file_combobox.set(self.image_files_in_current_test_set[self.current_image_file_index])

    def _clear_all_image_displays(self):
        for key in self.image_display_widgets:
            self.image_display_widgets[key].config(image=None, text="N/A")
            self.image_display_widgets[key].image = None
        self.image_pil_references.clear() # This stores the PhotoImageTk objects
        self.original_image_pil = {"pre": None, "post": None, "change": None} # Clear original PIL images too

    def _clear_image_file_list(self, message="No images"):
        self.image_file_combobox['values'] = [message]
        self.image_file_combobox.set(message)
        self.image_files_in_current_test_set = []
        self.current_image_file_index = -1
        self.update_navigation_buttons_state()

    def show_next_image(self):
        if not self.image_files_in_current_test_set: return
        self.current_image_file_index = (self.current_image_file_index + 1) % len(self.image_files_in_current_test_set)
        self.load_current_image_triplet()

    def show_previous_image(self):
        if not self.image_files_in_current_test_set: return
        self.current_image_file_index = (self.current_image_file_index - 1 + len(self.image_files_in_current_test_set)) % len(self.image_files_in_current_test_set)
        self.load_current_image_triplet()
        
    def update_navigation_buttons_state(self):
        num_img_files = len(self.image_files_in_current_test_set)
        if num_img_files <= 1:
            self.prev_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)
            self.next_button.config(state=tk.NORMAL)

    def _update_displayed_images(self):
        """
        Loads the PIL images into the display widgets, resizing them to fit.
        """
        for key, pil_image in self.original_image_pil.items():
            widget = self.image_display_widgets[key]
            
            if pil_image:
                # Resize the original image to fit the display widget
                pil_image.thumbnail((self.image_display_size[0], self.image_display_size[1]), Image.Resampling.LANCZOS)
                
                tk_image = ImageTk.PhotoImage(pil_image)
                widget.config(image=tk_image)
                
                # IMPORTANT: Keep a reference to the PhotoImage to prevent garbage collection
                self.image_pil_references[key] = tk_image 
            else:
                # Clear the image if the source PIL image is None
                widget.config(image=None, text="N/A")
                if key in self.image_pil_references:
                    del self.image_pil_references[key]

    # --- NEW MAGNIFIER METHODS ---
    def _create_magnifier(self, event):
        if self.magnifier_window: # Already exists
            return
        
        self.magnifier_window = tk.Toplevel(self.master)
        # Make it a borderless window
        self.magnifier_window.overrideredirect(True)
        # Ensure it stays on top
        self.magnifier_window.wm_attributes("-topmost", True)

        # Create a frame to hold the three magnified views
        main_frame = ttk.Frame(self.magnifier_window, borderwidth=2, relief="solid")
        main_frame.pack()

        self.magnifier_labels = {}
        for key in ["pre", "post", "change"]:
            frame = ttk.Frame(main_frame)
            frame.pack(side=tk.LEFT, padx=1, pady=1)
            label = ttk.Label(frame, background="black")
            label.pack()
            self.magnifier_labels[key] = label

        # Initially position it to trigger the update
        self._update_magnifier(event)

    def _destroy_magnifier(self, event):
        if self.magnifier_window:
            self.magnifier_window.destroy()
            self.magnifier_window = None
            self.magnifier_labels = None

    def _update_magnifier(self, event):
        if not self.magnifier_window:
            return

        # Position the magnifier window near the cursor
        # The offset ensures the cursor isn't directly over the magnifier
        win_x = event.x_root + 20
        win_y = event.y_root + 20
        self.magnifier_window.geometry(f"+{win_x}+{win_y}")
        
        # Calculate the source region to crop from the original images
        widget = event.widget
        
        for key, original_pil in self.original_image_pil.items():
            if not original_pil:
                continue # Skip if image is not loaded

            # Determine which display widget triggered the event to get correct coordinates
            display_widget = self.image_display_widgets[key]
            
            # We need to map the cursor position on the *displayed* (potentially resized) image
            # back to the coordinate system of the *original* full-resolution image.
            
            # 1. Get cursor coordinates relative to the widget that triggered the event
            x_on_widget = event.x
            y_on_widget = event.y
            
            # 2. Get the size of the displayed image (it's not the widget size, but the thumbnail size)
            displayed_image_ref = self.image_pil_references.get(key)
            if not displayed_image_ref: continue
            
            displayed_w = displayed_image_ref.width()
            displayed_h = displayed_image_ref.height()
            
            # 3. Handle padding: Center the image if the widget is larger than the thumbnail
            pad_x = (display_widget.winfo_width() - displayed_w) // 2
            pad_y = (display_widget.winfo_height() - displayed_h) // 2
            
            # Adjust cursor position to be relative to the top-left of the actual image thumbnail
            x = x_on_widget - pad_x
            y = y_on_widget - pad_y
            
            # Clamp coordinates to be within the image bounds
            x = max(0, min(x, displayed_w))
            y = max(0, min(y, displayed_h))
            
            # 4. Calculate proportional position on the original image
            original_w, original_h = original_pil.size
            x_prop = x / displayed_w
            y_prop = y / displayed_h
            
            center_x_on_original = original_w * x_prop
            center_y_on_original = original_h * y_prop
            
            # 5. Define the crop box for the magnifier on the original image
            # The size of the region to sample from the original image
            crop_width = original_w / self.magnification_factor 
            crop_height = original_h / self.magnification_factor

            left = center_x_on_original - crop_width / 2
            top = center_y_on_original - crop_height / 2
            
            # --- Boundary Correction ---
            # Now, adjust the box to ensure it's fully inside the original image.
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            if left + crop_width > original_w:
                left = original_w - crop_width
            if top + crop_height > original_h:
                top = original_h - crop_height
            
            right = left + crop_width
            bottom = top + crop_height
            
            # 6. Crop, resize, and display
            # Ensure the box coordinates are integers for cropping
            crop_box = (int(left), int(top), int(right), int(bottom))
            cropped_pil = original_pil.crop(crop_box)
            magnified_pil = cropped_pil.resize((self.magnifier_size, self.magnifier_size), Image.Resampling.NEAREST) # Use NEAREST for sharp pixels
            
            tk_image = ImageTk.PhotoImage(magnified_pil)
            
            self.magnifier_labels[key].config(image=tk_image)
            self.magnifier_labels[key].image = tk_image # Keep reference

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageTripletVisualizer(root)
    root.geometry("1000x500") 
    root.mainloop() 