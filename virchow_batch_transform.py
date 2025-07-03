#!/usr/bin/env python3
"""
Virchow Color Jitter Batch Processing Script

This script applies Virchow-2 style color jitter augmentation to a batch of histopathology 
images and generates side-by-side comparison visualizations. It follows the same pattern 
as other augmentation test scripts in the digital pathology research toolkit.

SCIENTIFIC BACKGROUND:
----------------------
Implements pathology-aware color augmentation inspired by "Virchow 2: Scaling Self-Supervised 
Mixed-Magnification Models in Pathology" (Zimmermann et al., 2024). The augmentation pipeline 
includes:

1. Brightness/Contrast adjustment - Scanner calibration variation simulation
2. HSV jitter - Color harmony preservation  
3. Stain vector jitter - H&E staining protocol variation (Macenko deconvolution)
4. JPEG compression - Clinical workflow artifact simulation
5. LAB illumination shift - Microscope illumination variation

USAGE:
------
1. Place original histopathology images in 'test_virchow_color_jitter/' directory
2. Run: python3 virchow_batch_transform.py
3. Results (side-by-side comparisons) saved in 'test_virchow_color_jitter/results/'

IMPLEMENTATION APPROACH:
------------------------
Uses the VirchowColorJitter class from virchow_color_jitter.py for consistent augmentation
that preserves pathological features while simulating realistic inter-laboratory variations.

Author: Ryan Rodriguez
Date: 2024
License: MIT
References: Zimmermann, E., et al. "Virchow2: Scaling Self-Supervised Mixed-Magnification 
           Models in Pathology." arXiv preprint arXiv:2408.00738v3 (2024).
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from virchow_color_jitter import VirchowColorJitter


def create_results_directory(input_dir: str) -> str:
    """
    Create results subdirectory within the input directory.
    
    Parameters
    ----------
    input_dir : str
        Path to input directory containing original images
        
    Returns
    -------
    str
        Path to created results directory
    """
    results_dir = os.path.join(input_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"✓ Results directory ready: {results_dir}")
    return results_dir


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file with proper color space conversion.
    
    Parameters
    ----------
    image_path : str
        Path to input image file
        
    Returns
    -------
    np.ndarray
        RGB image array with shape (H, W, 3) and dtype uint8
        
    Raises
    ------
    ValueError
        If image cannot be loaded or is invalid
    """
    # Load image using OpenCV (BGR format)
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for proper color processing
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb


def apply_virchow_augmentation(image: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply Virchow-2 style color jitter augmentation to input image.
    
    Parameters
    ----------
    image : np.ndarray
        Input RGB image with shape (H, W, 3) and dtype uint8
    seed : int, optional
        Random seed for reproducible augmentation
        
    Returns
    -------
    np.ndarray
        Augmented RGB image with same shape and dtype as input
    """
    # Set random seed for reproducible results if specified
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize Virchow color jitter with default pathology-optimized parameters
    # These parameters are calibrated for realistic H&E staining variations
    transform = VirchowColorJitter(
        brightness_range=(0.9, 1.1),        # Conservative brightness variation
        hue_range=5,                        # Subtle hue shifts for H&E preservation
        saturation_range=5,                 # Moderate saturation changes
        stain_rotation_range=5.0,           # Realistic stain vector rotation
        stain_magnitude_range=0.05,         # Stain intensity scaling (±5%)
        jpeg_quality_range=(50, 100),       # Clinical compression range
        lab_l_range=(-15, 15),              # Illumination lightness variation
        lab_a_range=(-5, 5),                # Green-red balance shifts
        lab_b_range=(-5, 5),                # Blue-yellow balance shifts
        p_stain=0.8,                        # High probability for stain variation
        p_jpeg=0.5,                         # Medium probability for compression
        p_lab=0.7,                          # High probability for illumination shifts
    )
    
    # Apply the complete augmentation pipeline
    # Returns dictionary with 'image' key containing augmented result
    augmented_result = transform(image=image)
    augmented_image = augmented_result["image"]
    
    return augmented_image


def create_side_by_side_comparison(original: np.ndarray, augmented: np.ndarray, 
                                 filename: str, output_path: str) -> None:
    """
    Create and save side-by-side comparison of original and augmented images.
    
    Parameters
    ----------
    original : np.ndarray
        Original RGB image
    augmented : np.ndarray
        Augmented RGB image  
    filename : str
        Original filename (without extension) for labeling
    output_path : str
        Full path where comparison image should be saved
    """
    # Create figure with side-by-side subplots and more space for titles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display original image
    ax1.imshow(original)
    ax1.set_title('ORIGINAL IMAGE\n(Unmodified)', 
                  fontsize=16, fontweight='bold', 
                  color='darkblue', pad=20)
    ax1.axis('off')
    
    # Display augmented image  
    ax2.imshow(augmented)
    ax2.set_title('VIRCHOW AUGMENTED\n(Color Jitter Applied)', 
                  fontsize=16, fontweight='bold', 
                  color='darkred', pad=20)
    ax2.axis('off')
    
    # Add overall title with augmentation details
    fig.suptitle(f'Virchow-2 Color Jitter Comparison: {filename}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Add clear labels at the bottom of each image
    ax1.text(0.5, -0.05, 'BEFORE AUGMENTATION', 
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=14, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    ax2.text(0.5, -0.05, 'AFTER AUGMENTATION', 
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=14, fontweight='bold', color='darkred',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    # Add detailed description at the bottom
    fig.text(0.5, 0.02, 
             'Augmentation: Brightness/Contrast • HSV Jitter • Stain Vector Manipulation • JPEG Compression • LAB Illumination',
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Adjust layout to prevent overlap and provide more space
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.02, right=0.98)
    
    # Save with high quality for research documentation
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()  # Free memory
    
    print(f"  ✓ Saved comparison: {os.path.basename(output_path)}")


def process_batch_images(input_dir: str) -> None:
    """
    Process all images in the input directory with Virchow color jitter augmentation.
    
    Parameters
    ----------
    input_dir : str
        Directory containing input histopathology images
    """
    print("=" * 70)
    print("VIRCHOW COLOR JITTER BATCH PROCESSING")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    
    # Create results subdirectory
    results_dir = create_results_directory(input_dir)
    
    # Supported image formats for pathology images
    supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    # Find all image files in input directory
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in supported_formats]
    
    if not image_files:
        print(f"❌ No supported image files found in {input_dir}")
        print(f"   Supported formats: {', '.join(supported_formats)}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) to process")
    print(f"🔬 Applying Virchow-2 style color jitter augmentation...")
    print()
    
    # Process each image
    successful_count = 0
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            # Load original image
            original_image = load_image(str(image_file))
            print(f"  📐 Image size: {original_image.shape[1]}×{original_image.shape[0]} pixels")
            
            # Apply Virchow color jitter augmentation
            # Use file index as seed for reproducible results across runs
            augmented_image = apply_virchow_augmentation(original_image, seed=i*42)
            
            # Calculate augmentation statistics for quality validation
            pixel_diff = np.mean(np.abs(augmented_image.astype(float) - original_image.astype(float)))
            print(f"  📊 Average pixel change: {pixel_diff:.2f} intensity units")
            
            # Create output filename for comparison image
            base_name = image_file.stem  # Filename without extension
            output_filename = f"{base_name}_virchow_comparison.jpg"
            output_path = os.path.join(results_dir, output_filename)
            
            # Generate and save side-by-side comparison
            create_side_by_side_comparison(original_image, augmented_image, 
                                         base_name, output_path)
            
            successful_count += 1
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {image_file.name}: {e}")
            print()
            continue
    
    # Final summary
    print("=" * 70)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Successfully processed: {successful_count}/{len(image_files)} images")
    print(f"📂 Results saved in: {results_dir}")
    print()
    print("🔬 AUGMENTATION DETAILS:")
    print("  • Brightness/Contrast: ±10% variation (scanner calibration simulation)")
    print("  • HSV Jitter: ±5° hue, ±5% saturation/value (color harmony preservation)")
    print("  • Stain Vector Jitter: ±5° rotation, ±5% magnitude (H&E protocol variation)")
    print("  • JPEG Compression: 50-100% quality (clinical workflow simulation)")
    print("  • LAB Illumination: ±15 L*, ±5 a*/b* (microscope variation)")
    print()
    print("📊 RESEARCH APPLICATIONS:")
    print("  • Foundation model training with cross-institutional robustness")
    print("  • Domain adaptation across different pathology laboratories")
    print("  • Stain normalization algorithm development and validation")
    print("  • Robustness evaluation of existing diagnostic AI models")
    print()
    print("🎯 Next steps: Review comparison images to validate augmentation quality")


def main():
    """
    Main execution function for batch processing.
    """
    # Define input directory following the established naming convention
    input_directory = "test_virchow_color_jitter"
    
    # Verify input directory exists
    if not os.path.exists(input_directory):
        print(f"❌ Input directory not found: {input_directory}")
        print(f"   Please create the directory and add histopathology images for testing.")
        print(f"   Supported formats: .jpg, .jpeg, .png, .tiff, .tif, .bmp")
        return 1
    
    try:
        # Process all images in the input directory
        process_batch_images(input_directory)
        return 0
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 