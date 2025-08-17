import os
import cv2
import numpy as np
from tqdm import tqdm

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def split_image(image_path, output_dir, base_name):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error reading image: {image_path}")
        return 0
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Size of patches
    patch_size = 256
    
    # Number of patches in each dimension
    n_h = h // patch_size
    n_w = w // patch_size
    
    patch_count = 0
    # Extract and save patches
    for i in range(n_h):
        for j in range(n_w):
            # Extract patch
            y1 = i * patch_size
            y2 = y1 + patch_size
            x1 = j * patch_size
            x2 = x1 + patch_size
            
            patch = img[y1:y2, x1:x2]
            
            # Save patch
            patch_name = f"{base_name}_{i}_{j}.png"
            cv2.imwrite(os.path.join(output_dir, patch_name), patch)
            patch_count += 1
    
    return patch_count

def process_dataset(input_root, output_root):
    # Expected patch counts from the paper
    expected_counts = {
        'train': 7120,
        'val': 1024,
        'test': 2048
    }
    
    # Create output directories
    splits = ['train', 'val', 'test']
    subdirs = ['A', 'B', 'label']
    
    for split in splits:
        for subdir in subdirs:
            create_dir(os.path.join(output_root, split, subdir))
    
    # Process each split
    actual_counts = {}
    for split in splits:
        print(f"\nProcessing {split} split...")
        split_patch_count = 0
        
        # Process each type of image (A, B, label)
        for subdir in subdirs:
            input_dir = os.path.join(input_root, split, subdir)
            output_dir = os.path.join(output_root, split, subdir)
            
            # Get list of images
            images = [f for f in os.listdir(input_dir) if f.endswith('.png')]
            
            # Process each image
            patch_count = 0
            for img_name in tqdm(images, desc=f"Processing {split}/{subdir}"):
                base_name = os.path.splitext(img_name)[0]
                img_path = os.path.join(input_dir, img_name)
                patches = split_image(img_path, output_dir, base_name)
                patch_count += patches
            
            if subdir == 'A':  # Only count patches from A images
                split_patch_count = patch_count
                actual_counts[split] = patch_count
        
        print(f"\n{split} split statistics:")
        print(f"Expected patches: {expected_counts[split]}")
        print(f"Actual patches: {split_patch_count}")
        
        if split_patch_count != expected_counts[split]:
            print(f"WARNING: Patch count mismatch for {split} split!")
            print(f"Expected {expected_counts[split]} patches but got {split_patch_count} patches")

def main():
    # Define input and output paths using absolute path
    input_root = r"D:\CGNet-CD-main\CGNet-CD-main\dataset\LEVIR-CD"
    output_root = os.path.join(os.path.dirname(input_root), "LEVIR-CD-256")
    
    print("Starting dataset preprocessing...")
    print(f"Input dataset path: {input_root}")
    print(f"Output dataset path: {output_root}")
    
    process_dataset(input_root, output_root)
    print("\nDataset preprocessing completed!")

if __name__ == "__main__":
    main() 