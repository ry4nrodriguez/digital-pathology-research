#!/usr/bin/env python3
"""
NeuroPath Optimized Virchow Color Jitter Batch Processing Script

This script applies the OPTIMIZED NeuroPath-calibrated Virchow color jitter to the same
test images to allow direct comparison with the original implementation results.

COMPARISON PURPOSE:
-------------------
- Original parameters: brightness±10%, stain±5°, aggressive settings
- Optimized parameters: brightness±5%, stain±3°, neuropathology-tuned settings
- Expected outcome: Reduced pixel change from 90+ to 30-40 intensity units

Usage: python3 virchow_optimized_batch_transform.py

Results will be saved in test_virchow_color_jitter/optimized_results/ for comparison.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from virchow_color_jitter_optimized import NeuroPathVirchowColorJitter


def create_optimized_results_directory(input_dir: str) -> str:
    """Create optimized results subdirectory within the input directory."""
    results_dir = os.path.join(input_dir, 'optimized_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"✓ Optimized results directory ready: {results_dir}")
    return results_dir


def load_image(image_path: str) -> np.ndarray:
    """Load image from file with proper color space conversion."""
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB for proper color processing
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def apply_optimized_virchow_augmentation(image: np.ndarray, seed: int = None) -> np.ndarray:
    """Apply NeuroPath-optimized Virchow color jitter augmentation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize OPTIMIZED NeuroPath Virchow color jitter
    # These parameters are calibrated for neuropathology with reduced intensity
    transform = NeuroPathVirchowColorJitter(
        brightness_range=(0.95, 1.05),        # REDUCED: ±5% vs ±10%
        hue_range=3,                          # REDUCED: ±3° vs ±5°
        saturation_range=3,                   # REDUCED: ±3% vs ±5%
        stain_rotation_range=3.0,             # REDUCED: ±3° vs ±5°
        stain_magnitude_range=0.03,           # REDUCED: ±3% vs ±5%
        jpeg_quality_range=(70, 100),         # IMPROVED: 70-100% vs 50-100%
        lab_l_range=(-10, 10),                # REDUCED: ±10 vs ±15
        lab_a_range=(-3, 3),                  # REDUCED: ±3 vs ±5
        lab_b_range=(-3, 3),                  # REDUCED: ±3 vs ±5
        p_stain=0.7,                          # REDUCED: 70% vs 80%
        p_jpeg=0.3,                           # REDUCED: 30% vs 50%
        p_lab=0.6,                            # REDUCED: 60% vs 70%
    )
    
    # Apply the optimized augmentation pipeline
    augmented_result = transform(image=image)
    augmented_image = augmented_result["image"]
    
    return augmented_image


def create_optimized_comparison(original: np.ndarray, augmented: np.ndarray, 
                              filename: str, output_path: str, pixel_change: float) -> None:
    """Create and save side-by-side comparison with optimization indicators."""
    # Create figure with enhanced layout for comparison analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display original image
    ax1.imshow(original)
    ax1.set_title('ORIGINAL IMAGE\n(Unmodified)', 
                  fontsize=16, fontweight='bold', 
                  color='darkblue', pad=20)
    ax1.axis('off')
    
    # Display optimized augmented image  
    ax2.imshow(augmented)
    ax2.set_title('NEUROPATH OPTIMIZED\n(Calibrated Parameters)', 
                  fontsize=16, fontweight='bold', 
                  color='darkgreen', pad=20)
    ax2.axis('off')
    
    # Add overall title with optimization details
    fig.suptitle(f'NeuroPath Optimized Virchow Augmentation: {filename}', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Add clear labels with quantitative results
    ax1.text(0.5, -0.05, 'BEFORE AUGMENTATION', 
             transform=ax1.transAxes, ha='center', va='top',
             fontsize=14, fontweight='bold', color='darkblue',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Status indicator based on pixel change
    status_color = 'green' if 20 <= pixel_change <= 40 else 'orange' if pixel_change < 60 else 'red'
    status_text = 'OPTIMAL' if 20 <= pixel_change <= 40 else 'MODERATE' if pixel_change < 60 else 'HIGH'
    
    ax2.text(0.5, -0.05, f'AFTER OPTIMIZATION\nPixel Change: {pixel_change:.1f} ({status_text})', 
             transform=ax2.transAxes, ha='center', va='top',
             fontsize=14, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    # Add parameter comparison at the bottom
    param_text = ('OPTIMIZED PARAMETERS: Brightness±5% • Hue±3° • Stain±3° • LAB±10 • '
                 'Target: 20-40 intensity units (vs Original: 90+ units)')
    fig.text(0.5, 0.02, param_text,
             ha='center', fontsize=11, style='italic', color='darkgreen', weight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.02, right=0.98)
    
    # Save with high quality
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  ✓ Saved optimized comparison: {os.path.basename(output_path)}")


def process_optimized_batch_images(input_dir: str) -> None:
    """Process the same images with optimized parameters for direct comparison."""
    print("=" * 80)
    print("NEUROPATH OPTIMIZED VIRCHOW COLOR JITTER BATCH PROCESSING")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print("🎯 OPTIMIZATION GOAL: Reduce pixel change from 90+ to 20-40 intensity units")
    print()
    
    # Create optimized results subdirectory
    results_dir = create_optimized_results_directory(input_dir)
    
    # Find the same image files used in original testing
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) for optimization testing")
    print(f"🔬 Applying NeuroPath-optimized augmentation parameters...")
    print()
    
    # Track optimization results
    pixel_changes = []
    successful_count = 0
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Processing: {image_file.name}")
            
            # Load original image
            original_image = load_image(str(image_file))
            print(f"  📐 Image size: {original_image.shape[1]}×{original_image.shape[0]} pixels")
            
            # Apply optimized Virchow color jitter
            # Use same seed as original for direct comparison
            augmented_image = apply_optimized_virchow_augmentation(original_image, seed=i*42)
            
            # Calculate optimization metrics
            pixel_diff = np.mean(np.abs(augmented_image.astype(float) - original_image.astype(float)))
            pixel_changes.append(pixel_diff)
            
            # Status assessment
            status = "🟢 OPTIMAL" if 20 <= pixel_diff <= 40 else "🟡 MODERATE" if pixel_diff < 60 else "🔴 HIGH"
            print(f"  📊 Average pixel change: {pixel_diff:.2f} intensity units {status}")
            
            # Create output filename
            base_name = image_file.stem
            output_filename = f"{base_name}_optimized_comparison.jpg"
            output_path = os.path.join(results_dir, output_filename)
            
            # Generate optimized comparison image
            create_optimized_comparison(original_image, augmented_image, 
                                      base_name, output_path, pixel_diff)
            
            successful_count += 1
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing {image_file.name}: {e}")
            print()
            continue
    
    # Comprehensive optimization analysis
    print("=" * 80)
    print("OPTIMIZATION RESULTS ANALYSIS")
    print("=" * 80)
    print(f"✅ Successfully processed: {successful_count}/{len(image_files)} images")
    print(f"📂 Optimized results saved in: {results_dir}")
    print()
    
    if pixel_changes:
        avg_change = np.mean(pixel_changes)
        min_change = np.min(pixel_changes)
        max_change = np.max(pixel_changes)
        std_change = np.std(pixel_changes)
        
        print("📊 QUANTITATIVE OPTIMIZATION ASSESSMENT:")
        print(f"  • Average pixel change: {avg_change:.2f} intensity units")
        print(f"  • Range: {min_change:.2f} - {max_change:.2f} intensity units")
        print(f"  • Standard deviation: {std_change:.2f}")
        print(f"  • Target range: 20-40 intensity units")
        
        # Optimization success assessment
        optimal_count = sum(1 for change in pixel_changes if 20 <= change <= 40)
        moderate_count = sum(1 for change in pixel_changes if 40 < change < 60)
        high_count = sum(1 for change in pixel_changes if change >= 60)
        
        print()
        print("🎯 OPTIMIZATION SUCCESS RATE:")
        print(f"  🟢 Optimal (20-40 units): {optimal_count}/{len(pixel_changes)} ({optimal_count/len(pixel_changes)*100:.1f}%)")
        print(f"  🟡 Moderate (40-60 units): {moderate_count}/{len(pixel_changes)} ({moderate_count/len(pixel_changes)*100:.1f}%)")
        print(f"  🔴 High (60+ units): {high_count}/{len(pixel_changes)} ({high_count/len(pixel_changes)*100:.1f}%)")
        
        print()
        print("📈 COMPARISON WITH ORIGINAL IMPLEMENTATION:")
        print(f"  • Original average: ~90-126 intensity units")
        print(f"  • Optimized average: {avg_change:.2f} intensity units")
        print(f"  • Improvement: {((90-avg_change)/90)*100:.1f}% reduction in intensity")
        
        overall_status = "SUCCESS" if avg_change <= 50 else "PARTIAL" if avg_change <= 70 else "NEEDS_TUNING"
        print(f"  • Overall optimization: {overall_status}")
    
    print()
    print("🔬 PARAMETER OPTIMIZATION SUMMARY:")
    print("  • Brightness: ±10% → ±5% (50% reduction)")
    print("  • Hue shifts: ±5° → ±3° (40% reduction)")
    print("  • Stain rotation: ±5° → ±3° (40% reduction)")
    print("  • LAB illumination: ±15 → ±10 (33% reduction)")
    print("  • Probabilities: Reduced by 10-20% across all transforms")
    
    print()
    print("🎯 Next steps:")
    print("  1. Compare optimized_results/ with original results/")
    print("  2. Evaluate neuropathology feature preservation")
    print("  3. Test on GM/WM/Meninge classification accuracy")
    print("  4. Further fine-tune if needed for <40 intensity units")


def main():
    """Main execution function for optimized batch processing."""
    input_directory = "test_virchow_color_jitter"
    
    # Verify input directory exists
    if not os.path.exists(input_directory):
        print(f"❌ Input directory not found: {input_directory}")
        return 1
    
    try:
        # Process with optimized parameters
        process_optimized_batch_images(input_directory)
        return 0
        
    except Exception as e:
        print(f"❌ Optimized batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 