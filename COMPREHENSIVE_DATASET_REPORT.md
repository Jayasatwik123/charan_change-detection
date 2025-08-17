# COMPREHENSIVE DATASET ANALYSIS REPORT FOR CGNET PROJECT

## Overview
This report provides an in-depth analysis of all datasets used in the CGNet (Change Guiding Network) project for remote sensing change detection.

## Dataset Summary

### 1. LEVIR-CD-256 Dataset
**Description:** Large-scale Building Change Detection dataset with 256x256 pixel patches
- **Total Images:** 10,192 image triplets (A, B, label)
- **Image Dimensions:** 256 × 256 pixels
- **Image Format:** RGB for images, Binary for labels
- **Training Set:** 7,120 image triplets
- **Validation Set:** 1,024 image triplets  
- **Test Set:** 2,048 image triplets

**Dataset Structure:**
```
LEVIR-CD-256/
├── train/
│   ├── A/          # Pre-change images (7,120 images)
│   ├── B/          # Post-change images (7,120 images)
│   └── label/      # Ground truth masks (7,120 images)
├── val/
│   ├── A/          # Pre-change images (1,024 images)
│   ├── B/          # Post-change images (1,024 images)
│   └── label/      # Ground truth masks (1,024 images)
└── test/
    ├── A/          # Pre-change images (2,048 images)
    ├── B/          # Post-change images (2,048 images)
    └── label/      # Ground truth masks (2,048 images)
```

**Sample Filenames:** 
- Images: `train_100_0_0.png`, `val_10_0_0.png`, `test_100_0_0.png`
- Naming convention indicates patch extraction from larger images

### 2. SYSU-CD Dataset
**Description:** Very High Resolution (VHR) change detection dataset from Hong Kong (2007-2014)
- **Total Images:** 20,000 image triplets (A, B, label)
- **Image Dimensions:** 256 × 256 pixels
- **Image Format:** RGB for images, Binary for labels
- **Training Set:** 12,000 image triplets
- **Validation Set:** 4,000 image triplets
- **Test Set:** 4,000 image triplets

**Dataset Structure:**
```
SYSU-CD/
├── list/           # Additional metadata files
├── train/
│   ├── A/                      # Pre-change images (12,000 images)
│   ├── B/                      # Post-change images (12,000 images)
│   ├── label/                  # Ground truth masks (12,000 images)
│   ├── A_clipcls_56_vit16.json # CLIP classification metadata
│   └── B_clipcls_56_vit16.json # CLIP classification metadata
├── val/
│   ├── A/          # Pre-change images (4,000 images)
│   ├── B/          # Post-change images (4,000 images)
│   └── label/      # Ground truth masks (4,000 images)
├── test/
│   ├── A/          # Pre-change images (4,000 images)
│   ├── B/          # Post-change images (4,000 images)
│   └── label/      # Ground truth masks (4,000 images)
├── train.txt       # Training file paths (12,001 lines)
├── val.txt         # Validation file paths
└── test.txt        # Test file paths
```

**Sample Filenames:**
- Images: `00000.png`, `00001.png`, `03996.png`
- Sequential numeric naming convention

## Image Specifications

### Image Properties
- **Resolution:** 256 × 256 pixels for all datasets
- **Color Mode:** RGB (3 channels) for input images
- **Label Mode:** Binary (grayscale) for ground truth masks
- **File Format:** PNG
- **Label Values:** 
  - 0 (black): No change pixels
  - 255 (white): Change pixels

### Preprocessing Pipeline

#### 1. Data Augmentation (Training Only)
**Standard Augmentations:**
- **Horizontal Flip:** Random left-right flip (50% probability)
- **Random Crop:** Border crop with 30-pixel margin
- **Random Rotation:** ±15 degrees (20% probability)
- **Color Enhancement:**
  - Brightness: 0.5-1.5 factor
  - Contrast: 0.5-1.5 factor  
  - Color saturation: 0.0-2.0 factor
  - Sharpness: 0.0-3.0 factor

**Advanced Augmentations:**
- **Vertical Flip:** Random top-bottom flip (50% probability)
- **90-degree Rotation:** Random 90°, 180°, 270° rotation (30% probability)
- **Salt & Pepper Noise:** Random pixel corruption
- **Mosaic Augmentation:** Multi-image composition (75% probability)

#### 2. Normalization
**Image Transform Pipeline:**
```python
transforms.Compose([
    transforms.Resize((trainsize, trainsize)),  # Default: 256x256
    transforms.ToTensor(),                      # Convert to tensor [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1,1]
])
```

**Label Transform Pipeline:**
```python
transforms.Compose([
    transforms.Resize((trainsize, trainsize)),  # Default: 256x256
    transforms.ToTensor()                       # Convert to tensor [0,1]
])
```

## Dataset Usage in CGNet

### Training Configuration
- **Batch Size:** Configurable (typical: 8-16)
- **Input Size:** 256 × 256 pixels
- **Data Parallel:** Multi-GPU support
- **Workers:** Multi-threaded data loading

### Data Loading Strategy
1. **File Discovery:** Automatic scanning of A/, B/, label/ folders
2. **File Filtering:** Validation of matching triplets
3. **Sequential Loading:** Sorted filename processing
4. **Memory Management:** Efficient PIL image handling

### Dataset Selection
The training script supports dynamic dataset selection:
- Automatic detection of available datasets in `dataset/` folder
- Interactive selection during training startup
- Support for multiple datasets in same project structure

## Preprocessing Scripts

### 1. LEVIR-CD Preprocessing (`preprocess_dataset.py`)
- **Purpose:** Convert large LEVIR-CD images to 256×256 patches
- **Input:** Original LEVIR-CD dataset with large images
- **Output:** LEVIR-CD-256 with standardized patches
- **Patch Generation:** Non-overlapping 256×256 windows
- **Expected Counts:** Validates against known patch numbers

### 2. S2Looking Preprocessing (`preprocess_s2looking.py`)  
- **Purpose:** Process S2Looking dataset for change detection
- **Features:** Handles challenging imbalanced datasets
- **Patch Size:** Configurable (default: 256×256)

## Storage and Organization

### File System Layout
```
dataset/
├── LEVIR-CD-256/     # 10,192 triplets
│   ├── train/        # 7,120 triplets  
│   ├── val/          # 1,024 triplets
│   └── test/         # 2,048 triplets
└── SYSU-CD/          # 20,000 triplets
    ├── train/        # 12,000 triplets
    ├── val/          # 4,000 triplets
    └── test/         # 4,000 triplets
```

### Storage Requirements
- **LEVIR-CD-256:** ~1.5GB (10,192 × 3 × 256×256×3 bytes)
- **SYSU-CD:** ~3.0GB (20,000 × 3 × 256×256×3 bytes)
- **Total:** ~4.5GB for both datasets

## Additional Features

### SYSU-CD Enhancements
- **CLIP Integration:** Pre-computed CLIP-ViT classifications
- **Metadata Files:** JSON files with semantic classifications
- **Scene Categories:** Stadium, river, bridge, industrial area, etc.
- **Confidence Scores:** Numerical confidence for each category

### Quality Assurance
- **Size Validation:** Ensures all triplet images match dimensions
- **File Integrity:** Checks for corrupted or missing files
- **Format Consistency:** Validates PNG format and color modes

## Performance Considerations

### Data Loading Optimization
- **Multi-threading:** Parallel data loading workers
- **Memory Pinning:** GPU transfer optimization
- **Batch Processing:** Efficient batch formation
- **Caching Strategy:** PIL image object management

### Training Efficiency
- **Mosaic Augmentation:** Reduces training time while maintaining diversity
- **Random Sampling:** Balanced dataset utilization
- **GPU Acceleration:** CUDA-optimized tensor operations

## Conclusion

The CGNet project utilizes two high-quality, well-preprocessed datasets totaling 30,192 image triplets. Both datasets provide 256×256 pixel RGB images with binary change masks, suitable for deep learning change detection models. The comprehensive preprocessing pipeline includes extensive data augmentation and normalization to improve model robustness and generalization.
