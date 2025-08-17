import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def create_dir_if_not_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def split_image_into_patches(img_path, output_dir, patch_size=256):
    """Split a single image into non-overlapping patches"""
    try:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return 0
        
        height, width = img.shape[:2]
        patches_created = 0
        
        # Calculate number of patches in each dimension
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Extract patches
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                # Extract patch
                start_h = i * patch_size
                start_w = j * patch_size
                patch = img[start_h:start_h + patch_size, start_w:start_w + patch_size]
                
                # Create patch filename
                patch_filename = f"{base_name}_patch_{i}_{j}.png"
                patch_path = os.path.join(output_dir, patch_filename)
                
                # Save patch
                cv2.imwrite(patch_path, patch)
                patches_created += 1
        
        return patches_created
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return 0

def process_directory(input_dir, output_dir, patch_size=256):
    """Process all images in a directory"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    create_dir_if_not_exists(output_dir)
    
    # Get all PNG files
    image_files = list(input_dir.glob('*.png'))
    
    # Process each image with progress bar
    total_patches = 0
    with tqdm(total=len(image_files), desc=f"Processing {input_dir.name}") as pbar:
        for img_path in image_files:
            patches_created = split_image_into_patches(img_path, output_dir, patch_size)
            total_patches += patches_created
            pbar.update(1)
    
    return total_patches, len(image_files)

def main():
    # Base directories
    base_dir = r"C:\Users\Cynix\Desktop\CGNet_workingOnBackBone_ssd\dataset\S2Looking"
    output_base = os.path.join(base_dir, "256x256_patches")
    
    # Expected patch counts based on paper
    expected_counts = {
        "train": 56000,  # 3500 original pairs -> 56000 patch pairs
        "val": 8000,     # 500 original pairs -> 8000 patch pairs
        "test": 16000    # 1000 original pairs -> 16000 patch pairs
    }
    
    # Directories to process
    dirs_to_process = [
        ("train/A", "train/A"),
        ("train/B", "train/B"),
        ("train/label", "train/label"),
        ("val/A", "val/A"),
        ("val/B", "val/B"),
        ("val/label", "val/label"),
        ("test/A", "test/A"),
        ("test/B", "test/B"),
        ("test/label", "test/label")
    ]
    
    # Create the output base directory
    create_dir_if_not_exists(output_base)
    
    # Process each directory
    total_patches = 0
    total_images = 0
    
    print("\nStarting S2Looking dataset preprocessing (splitting into 256x256 patches)...")
    print("="*80)
    print("Expected patch counts from paper:")
    print(f"Training: {expected_counts['train']} patches (from 3500 image pairs)")
    print(f"Validation: {expected_counts['val']} patches (from 500 image pairs)")
    print(f"Testing: {expected_counts['test']} patches (from 1000 image pairs)")
    print("="*80)
    
    for input_subdir, output_subdir in dirs_to_process:
        input_path = os.path.join(base_dir, input_subdir)
        output_path = os.path.join(output_base, output_subdir)
        
        # Skip if input directory doesn't exist
        if not os.path.exists(input_path):
            print(f"Warning: Input directory not found: {input_path}")
            continue
        
        # Create output subdirectories
        create_dir_if_not_exists(os.path.dirname(output_path))
        
        # Process the directory
        patches_created, images_processed = process_directory(input_path, output_path)
        total_patches += patches_created
        total_images += images_processed
        
        # Get the split type (train/val/test)
        split_type = input_subdir.split('/')[0]
        expected = expected_counts.get(split_type, 0) // 3  # Divide by 3 because we have A, B, and label
        
        print(f"Processed {input_subdir}:")
        print(f"  - Original images: {images_processed}")
        print(f"  - Patches created: {patches_created}")
        if expected > 0:
            print(f"  - Expected patches: {expected}")
            if patches_created != expected:
                print(f"  - WARNING: Patch count mismatch! Expected {expected}, got {patches_created}")
    
    print("\nPreprocessing Summary:")
    print("="*80)
    print(f"Total original images processed: {total_images}")
    print(f"Total patches created: {total_patches}")
    print(f"Output directory: {output_base}")
    print("="*80)

if __name__ == "__main__":
    main() 