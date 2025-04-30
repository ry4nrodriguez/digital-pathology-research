"""
Tissue Augmentation Script Ryan's Implementations

This script applies four augmentation techniques to the tissue dataset:
1. Reinhard Stain Normalization (from stain_norm.py)
2. HED color space augmentation (from hed_transform.py)
3. Regional Gaussian Blur (from blur_transform.py)
4. Global Gaussian Noise (from noise_blur_transforms.py)

The augmented images are saved in the corresponding directories.
"""

import os
import cv2
import numpy as np
import random
import torch
import shutil
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import torch.nn.functional as F
from pathlib import Path

# Paths - Using relative paths for local implementation
SOURCE_DIR = "./tissue_type"
TARGET_DIR = "./augmented_tissue_dataset"

# Create main directories if they don't exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Number of images to process per tissue type (set to None for all images)
MAX_IMAGES = 100  # Set to None to process all images

# ================================================
# Reinhard Color Normalization Implementation
# ================================================
def reinhard_normalization(source_img, target_img=None):
    """
    Apply Reinhard color normalization to match source to target.
    Implementation from your stain_norm.py
    """
    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if target_img is None:
        # Default reference statistics based on high-quality samples
        target_mean = np.array([150, 135, 140], dtype=np.float32)
        target_std = np.array([30, 8, 8], dtype=np.float32)
    else:
        # Calculate statistics from target image
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
    
    # Calculate source statistics
    source_mean = np.mean(source_lab, axis=(0, 1))
    source_std = np.std(source_lab, axis=(0, 1))
    
    # Avoid division by zero
    source_std = np.clip(source_std, 0.0001, None)
    
    # Normalize the LAB image
    normalized_lab = ((source_lab - source_mean) * (target_std / source_std)) + target_mean
    
    # Clip to valid LAB range and convert back to uint8
    normalized_lab = np.clip(normalized_lab, 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)
    
    return normalized_bgr

# ================================================
# HED Transform Implementation
# ================================================
def hed_augmentation(img):
    """
    Apply HED transform based on your implementation from hed_transform.py
    """
    # Convert BGR to RGB for consistency with PyTorch
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PyTorch tensor (N, C, H, W)
    h, w, c = img_rgb.shape
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    # Define the Ruifrock matrix for HED color deconvolution
    ruifrock = torch.tensor(np.array([
        [0.65, 0.70, 0.29],
        [0.07, 0.99, 0.11],
        [0.27, 0.57, 0.78]
    ], dtype=np.float64))
    
    # Normalize the matrix
    ruifrock_normalized = F.normalize(ruifrock).float()
    
    # Random alpha and beta parameters for each H/E/D channel
    alpha = torch.tensor([[random.uniform(0.8, 1.2) for _ in range(3)]])
    beta = torch.tensor([[random.uniform(-0.2, 0.2) for _ in range(3)]])
    
    max_input_value = 255.0
    
    # Convert to normalized float
    x = (img_tensor / max_input_value).float()
    
    # Perform HED color deconvolution
    # Unmixing step
    x_copy = x.clone()
    x_hwc = x_copy.squeeze(0).permute(1, 2, 0)  # (H, W, C)
    x_hwc = torch.max(x_hwc, 1e-6 * torch.ones_like(x_hwc))
    Z = torch.log(x_hwc) / np.log(1e-6) @ torch.inverse(ruifrock_normalized)
    
    # Apply alpha and beta transformations
    for j, channel in enumerate(range(3)):
        Z[:, :, j] = (alpha[0, j] * Z[:, :, j]) + beta[0, j]
    
    # Remixing step
    log_adjust = -np.log(1E-6)
    log_rgb = -(Z * log_adjust) @ ruifrock_normalized
    rgb = torch.exp(log_rgb)
    rgb = torch.clamp(rgb, 0, 1)
    
    # Convert back to proper tensor format
    result = rgb.permute(2, 0, 1).unsqueeze(0)
    result = (result * max_input_value).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).numpy()
    
    # Convert RGB back to BGR for OpenCV
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    
    return result_bgr

# ================================================
# Regional Blur Transform Implementation
# ================================================
def apply_regional_blur(img, corner_x=50, corner_y=50, 
                       blur_height=100, blur_width=100,
                       kernel_size=5, sigma=10):
    """
    Apply regional blur based on your implementation from blur_transform.py
    """
    # Create a copy of the image
    img_blurred = img.copy()
    
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Extract the region to blur
    region = img[corner_y:corner_y+blur_height, corner_x:corner_x+blur_width]
    
    # Apply Gaussian blur to the region
    blurred_region = cv2.GaussianBlur(region, (kernel_size, kernel_size), sigma)
    
    # Replace the region in the image
    img_blurred[corner_y:corner_y+blur_height, corner_x:corner_x+blur_width] = blurred_region
    
    return img_blurred

# ================================================
# Gaussian Noise Implementation
# ================================================
def apply_gaussian_noise(img, mean=0, std=25):
    """
    Apply Gaussian noise based on your implementation from noise_blur_transforms.py
    """
    # Convert to float32 for arithmetic operations
    img_float = img.astype(np.float32)
    
    # Generate Gaussian noise
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    
    # Add noise to the image
    img_noisy = img_float + noise
    
    # Ensure values stay in the valid range
    img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
    
    return img_noisy

# Function to process a single image with all augmentations
def process_image(args):
    img_path, tissue_type = args
    try:
        # Read the image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return False
        
        # Get filename without extension
        filename = os.path.basename(img_path)
        
        # Apply Reinhard normalization (your implementation)
        img_reinhard = reinhard_normalization(img)
        
        # Apply HED augmentation (your implementation)
        img_hed = hed_augmentation(img)
        
        # Apply Regional Gaussian blur (your implementation)
        img_blur = apply_regional_blur(img, corner_x=50, corner_y=50, 
                                      blur_height=100, blur_width=100,
                                      kernel_size=5, sigma=10)
        
        # Apply Gaussian noise (your implementation)
        img_noise = apply_gaussian_noise(img, mean=0, std=25)
        
        # Save all augmentations to their respective directories
        cv2.imwrite(os.path.join(TARGET_DIR, tissue_type, "reinhard", filename), img_reinhard)
        cv2.imwrite(os.path.join(TARGET_DIR, tissue_type, "hed", filename), img_hed)
        cv2.imwrite(os.path.join(TARGET_DIR, tissue_type, "blur", filename), img_blur)
        cv2.imwrite(os.path.join(TARGET_DIR, tissue_type, "noise", filename), img_noise)
        
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    # Tissue types to process
    tissue_types = ["GM", "WM", "meninge"]
    
    for tissue_type in tissue_types:
        print(f"Processing {tissue_type} images...")
        
        # Create augmentation subdirectories
        for aug_type in ["reinhard", "hed", "blur", "noise"]:
            os.makedirs(os.path.join(TARGET_DIR, tissue_type, aug_type), exist_ok=True)
        
        # Get all image paths for this tissue type
        img_paths = glob.glob(os.path.join(SOURCE_DIR, tissue_type, "*.png"))
        
        # Limit the number of images if MAX_IMAGES is set
        if MAX_IMAGES:
            img_paths = img_paths[:MAX_IMAGES]
        
        print(f"Found {len(img_paths)} images in {tissue_type}")
        
        # Create arguments for parallel processing
        args = [(img_path, tissue_type) for img_path in img_paths]
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(process_image, args), total=len(args)))
        
        success_count = results.count(True)
        print(f"Successfully processed {success_count} out of {len(img_paths)} images for {tissue_type}")

if __name__ == "__main__":
    main()
    print("Augmentation complete!")
