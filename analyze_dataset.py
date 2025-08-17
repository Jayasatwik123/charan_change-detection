import os
from PIL import Image

def analyze_dataset():
    base_path = r"c:\Users\Cynix\Desktop\Sri charan working on this folder\CGNet_workingOnBackBone_ssd-done_workingOnChangingAttenionModule\dataset"
    datasets = ['LEVIR-CD-256', 'SYSU-CD']
    
    print("=== DATASET ANALYSIS REPORT ===\n")
    
    for dataset in datasets:
        print(f"=== {dataset} ===")
        dataset_path = os.path.join(base_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset path not found: {dataset_path}")
            continue
            
        total_images = 0
        
        for split in ['train', 'val', 'test']:
            split_path = os.path.join(dataset_path, split)
            
            if os.path.exists(split_path):
                print(f"\n{split.upper()} SET:")
                
                for folder in ['A', 'B', 'label']:
                    folder_path = os.path.join(split_path, folder)
                    
                    if os.path.exists(folder_path):
                        files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                        count = len(files)
                        print(f"  {folder}: {count} images")
                        
                        if folder == 'A':  # Count only once per triplet
                            total_images += count
                            
                        # Analyze first image for dimensions
                        if count > 0 and folder == 'A':
                            sample_path = os.path.join(folder_path, files[0])
                            try:
                                img = Image.open(sample_path)
                                print(f"  Sample image: {files[0]}")
                                print(f"  Dimensions: {img.size[0]}x{img.size[1]} pixels")
                                print(f"  Mode: {img.mode}")
                            except Exception as e:
                                print(f"  Error reading sample: {e}")
        
        print(f"\nTOTAL IMAGES IN {dataset}: {total_images}")
        print("-" * 50)

if __name__ == "__main__":
    analyze_dataset()
