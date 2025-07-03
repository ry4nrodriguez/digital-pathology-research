#!/usr/bin/env python3
"""
Virchow Color Jitter Stage-by-Stage Analysis
Dr. Shuying Li's NeuroPath Project - Boston University

This script applies each augmentation stage individually to show the cumulative
effect and identify which specific augmentation component is causing excessive
pixel intensity changes.

AUGMENTATION PIPELINE STAGES:
1. Original Image (baseline)
2. Brightness/Contrast Adjustment
3. HSV Jitter (Hue/Saturation/Value)
4. Stain Vector Jitter (Macenko deconvolution)
5. JPEG Compression Simulation
6. LAB Illumination Shift

PURPOSE: Identify which stage causes >40 intensity unit changes for optimization
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from virchow_color_jitter import StainJitter, LABShift


def create_stage_analysis_directory(input_dir: str) -> str:
    """Create stage analysis results directory."""
    results_dir = os.path.join(input_dir, 'stage_analysis')
    os.makedirs(results_dir, exist_ok=True)
    print(f"✓ Stage analysis directory ready: {results_dir}")
    return results_dir


def load_image(image_path: str) -> np.ndarray:
    """Load image from file with proper color space conversion."""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def apply_stage_by_stage_augmentation(image: np.ndarray, seed: int = None, 
                                    use_optimized_params: bool = True) -> tuple:
    """
    Apply augmentation stage by stage and return all intermediate results.
    
    Returns:
    --------
    tuple: (stage_images, stage_names, pixel_changes, cumulative_changes)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Choose parameter set based on configuration
    if use_optimized_params:
        # OPTIMIZED PARAMETERS (NeuroPath-calibrated)
        brightness_range = (0.95, 1.05)
        hue_range = 3
        saturation_range = 3
        value_range = 3
        stain_rotation_range = 3.0
        stain_magnitude_range = 0.03
        jpeg_quality_range = (70, 100)
        lab_l_range = (-10, 10)
        lab_a_range = (-3, 3)
        lab_b_range = (-3, 3)
        p_stain = 0.7
        p_jpeg = 0.3
        p_lab = 0.6
        param_label = "OPTIMIZED"
    else:
        # ORIGINAL PARAMETERS (Standard Virchow)
        brightness_range = (0.9, 1.1)
        hue_range = 5
        saturation_range = 5
        value_range = 5
        stain_rotation_range = 5.0
        stain_magnitude_range = 0.05
        jpeg_quality_range = (50, 100)
        lab_l_range = (-15, 15)
        lab_a_range = (-5, 5)
        lab_b_range = (-5, 5)
        p_stain = 0.8
        p_jpeg = 0.5
        p_lab = 0.7
        param_label = "ORIGINAL"
    
    # Initialize stage tracking
    stage_images = []
    stage_names = []
    pixel_changes = []
    cumulative_changes = []
    
    # Stage 0: Original Image (baseline)
    current_image = image.copy()
    stage_images.append(current_image.copy())
    stage_names.append("Original\n(Baseline)")
    pixel_changes.append(0.0)
    cumulative_changes.append(0.0)
    
    # Stage 1: Brightness/Contrast Adjustment
    try:
        brightness_transform = A.RandomBrightnessContrast(
            brightness_limit=abs(brightness_range[0] - 1.0),
            contrast_limit=abs(brightness_range[0] - 1.0),
            brightness_by_max=False,
            p=1.0,
        )
        result = brightness_transform(image=current_image)
        new_image = result["image"]
        
        stage_change = np.mean(np.abs(new_image.astype(float) - current_image.astype(float)))
        cumulative_change = np.mean(np.abs(new_image.astype(float) - image.astype(float)))
        
        stage_images.append(new_image.copy())
        stage_names.append(f"Brightness/Contrast\n(±{abs(brightness_range[0]-1.0)*100:.0f}%)")
        pixel_changes.append(stage_change)
        cumulative_changes.append(cumulative_change)
        current_image = new_image
        
    except Exception as e:
        print(f"Warning: Brightness/Contrast stage failed: {e}")
        stage_images.append(current_image.copy())
        stage_names.append("Brightness/Contrast\n(FAILED)")
        pixel_changes.append(0.0)
        cumulative_changes.append(cumulative_changes[-1])
    
    # Stage 2: HSV Jitter
    try:
        hsv_transform = A.HueSaturationValue(
            hue_shift_limit=hue_range,
            sat_shift_limit=saturation_range,
            val_shift_limit=value_range,
            p=1.0,
        )
        result = hsv_transform(image=current_image)
        new_image = result["image"]
        
        stage_change = np.mean(np.abs(new_image.astype(float) - current_image.astype(float)))
        cumulative_change = np.mean(np.abs(new_image.astype(float) - image.astype(float)))
        
        stage_images.append(new_image.copy())
        stage_names.append(f"HSV Jitter\n(H±{hue_range}°, S/V±{saturation_range}%)")
        pixel_changes.append(stage_change)
        cumulative_changes.append(cumulative_change)
        current_image = new_image
        
    except Exception as e:
        print(f"Warning: HSV stage failed: {e}")
        stage_images.append(current_image.copy())
        stage_names.append("HSV Jitter\n(FAILED)")
        pixel_changes.append(0.0)
        cumulative_changes.append(cumulative_changes[-1])
    
    # Stage 3: Stain Vector Jitter
    try:
        # Apply stain jitter with specified probability
        if np.random.random() < p_stain:
            stain_transform = StainJitter(
                rotation_range=stain_rotation_range,
                magnitude_range=stain_magnitude_range,
                p=1.0,  # Always apply when selected
            )
            result = stain_transform(image=current_image)
            new_image = result["image"]
            
            stage_change = np.mean(np.abs(new_image.astype(float) - current_image.astype(float)))
            cumulative_change = np.mean(np.abs(new_image.astype(float) - image.astype(float)))
            
            stage_images.append(new_image.copy())
            stage_names.append(f"Stain Jitter\n(±{stain_rotation_range}°, ±{stain_magnitude_range*100:.0f}%)")
            pixel_changes.append(stage_change)
            cumulative_changes.append(cumulative_change)
            current_image = new_image
        else:
            # Stain jitter skipped due to probability
            stage_images.append(current_image.copy())
            stage_names.append(f"Stain Jitter\n(SKIPPED - p={p_stain})")
            pixel_changes.append(0.0)
            cumulative_changes.append(cumulative_changes[-1])
            
    except Exception as e:
        print(f"Warning: Stain jitter stage failed: {e}")
        stage_images.append(current_image.copy())
        stage_names.append("Stain Jitter\n(FAILED)")
        pixel_changes.append(0.0)
        cumulative_changes.append(cumulative_changes[-1])
    
    # Stage 4: JPEG Compression
    try:
        # Apply JPEG compression with specified probability
        if np.random.random() < p_jpeg:
            # Note: Using quality_lower/upper might not work, so use quality_range
            jpeg_transform = A.ImageCompression(
                quality_lower=jpeg_quality_range[0],
                quality_upper=jpeg_quality_range[1],
                p=1.0,
            )
            result = jpeg_transform(image=current_image)
            new_image = result["image"]
            
            stage_change = np.mean(np.abs(new_image.astype(float) - current_image.astype(float)))
            cumulative_change = np.mean(np.abs(new_image.astype(float) - image.astype(float)))
            
            stage_images.append(new_image.copy())
            stage_names.append(f"JPEG Compression\n({jpeg_quality_range[0]}-{jpeg_quality_range[1]}% quality)")
            pixel_changes.append(stage_change)
            cumulative_changes.append(cumulative_change)
            current_image = new_image
        else:
            # JPEG compression skipped due to probability
            stage_images.append(current_image.copy())
            stage_names.append(f"JPEG Compression\n(SKIPPED - p={p_jpeg})")
            pixel_changes.append(0.0)
            cumulative_changes.append(cumulative_changes[-1])
            
    except Exception as e:
        print(f"Warning: JPEG compression stage failed: {e}")
        stage_images.append(current_image.copy())
        stage_names.append("JPEG Compression\n(FAILED)")
        pixel_changes.append(0.0)
        cumulative_changes.append(cumulative_changes[-1])
    
    # Stage 5: LAB Illumination Shift
    try:
        # Apply LAB shift with specified probability
        if np.random.random() < p_lab:
            lab_transform = LABShift(
                l_range=lab_l_range,
                a_range=lab_a_range,
                b_range=lab_b_range,
                p=1.0,
            )
            result = lab_transform(image=current_image)
            new_image = result["image"]
            
            stage_change = np.mean(np.abs(new_image.astype(float) - current_image.astype(float)))
            cumulative_change = np.mean(np.abs(new_image.astype(float) - image.astype(float)))
            
            stage_images.append(new_image.copy())
            stage_names.append(f"LAB Illumination\n(L±{lab_l_range[1]}, a/b±{lab_a_range[1]})")
            pixel_changes.append(stage_change)
            cumulative_changes.append(cumulative_change)
            current_image = new_image
        else:
            # LAB shift skipped due to probability
            stage_images.append(current_image.copy())
            stage_names.append(f"LAB Illumination\n(SKIPPED - p={p_lab})")
            pixel_changes.append(0.0)
            cumulative_changes.append(cumulative_changes[-1])
            
    except Exception as e:
        print(f"Warning: LAB illumination stage failed: {e}")
        stage_images.append(current_image.copy())
        stage_names.append("LAB Illumination\n(FAILED)")
        pixel_changes.append(0.0)
        cumulative_changes.append(cumulative_changes[-1])
    
    return stage_images, stage_names, pixel_changes, cumulative_changes, param_label


def create_stage_progression_visualization(stage_images, stage_names, pixel_changes, 
                                         cumulative_changes, filename, output_path, param_label):
    """Create comprehensive stage-by-stage visualization."""
    
    num_stages = len(stage_images)
    
    # Create figure with improved spacing for clear title visibility
    fig = plt.figure(figsize=(20, 14))
    
    # Row 1: Stage progression images with clear spacing
    for i, (img, name, stage_change, cum_change) in enumerate(
        zip(stage_images, stage_names, pixel_changes, cumulative_changes)):
        
        ax = plt.subplot(4, num_stages, i + 1)
        ax.imshow(img)
        ax.set_title(f"{name}\nStage Δ: {stage_change:.1f}\nTotal Δ: {cum_change:.1f}", 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
        
        # Color-code based on change magnitude
        if stage_change > 40:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                     fill=False, edgecolor='red', linewidth=3))
        elif stage_change > 20:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                     fill=False, edgecolor='orange', linewidth=3))
        elif stage_change > 0:
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                     fill=False, edgecolor='green', linewidth=2))
    
    # Row 2: Stage-wise change analysis
    ax_stage = plt.subplot(4, 2, 3)
    bars = ax_stage.bar(range(len(pixel_changes)), pixel_changes, 
                       color=['gray'] + ['red' if x > 40 else 'orange' if x > 20 else 'green' 
                              for x in pixel_changes[1:]])
    ax_stage.set_title('Stage-wise Pixel Changes', fontweight='bold')
    ax_stage.set_xlabel('Augmentation Stage')
    ax_stage.set_ylabel('Pixel Intensity Change')
    ax_stage.set_xticks(range(len(stage_names)))
    ax_stage.set_xticklabels([name.split('\n')[0] for name in stage_names], rotation=45)
    ax_stage.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='High threshold')
    ax_stage.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Moderate threshold')
    ax_stage.legend()
    ax_stage.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, pixel_changes):
        if value > 0:
            ax_stage.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Row 3: Cumulative change analysis
    ax_cum = plt.subplot(4, 2, 4)
    ax_cum.plot(range(len(cumulative_changes)), cumulative_changes, 'b-o', linewidth=2, markersize=6)
    ax_cum.set_title('Cumulative Pixel Changes', fontweight='bold')
    ax_cum.set_xlabel('Augmentation Stage')
    ax_cum.set_ylabel('Total Pixel Intensity Change')
    ax_cum.set_xticks(range(len(stage_names)))
    ax_cum.set_xticklabels([name.split('\n')[0] for name in stage_names], rotation=45)
    ax_cum.axhline(y=40, color='red', linestyle='--', alpha=0.7, label='Target maximum (40)')
    ax_cum.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Target minimum (20)')
    ax_cum.fill_between(range(len(cumulative_changes)), 20, 40, alpha=0.2, color='green', label='Target range')
    ax_cum.legend()
    ax_cum.grid(True, alpha=0.3)
    
    # Add analysis text spanning the bottom row for better readability
    ax_text = plt.subplot(4, 1, 4)
    ax_text.axis('off')
    
    # Find the stage with maximum change
    max_stage_idx = np.argmax(pixel_changes[1:]) + 1  # Skip original image
    max_stage_name = stage_names[max_stage_idx].split('\n')[0]
    max_stage_change = pixel_changes[max_stage_idx]
    
    final_change = cumulative_changes[-1]
    target_status = "OPTIMAL" if 20 <= final_change <= 40 else "MODERATE" if final_change < 60 else "HIGH"
    
    analysis_text = f"""
STAGE-BY-STAGE ANALYSIS RESULTS ({param_label} PARAMETERS)
{'='*60}

🎯 TARGET: 20-40 total intensity units for neuropathology
📊 ACTUAL: {final_change:.1f} intensity units ({target_status})

🔍 STAGE WITH MAXIMUM IMPACT:
   • Stage: {max_stage_name}
   • Change: {max_stage_change:.1f} intensity units
   • Percentage of total: {(max_stage_change/final_change)*100:.1f}%

📈 STAGE BREAKDOWN:
"""
    for i, (name, stage_change, cum_change) in enumerate(
        zip(stage_names[1:], pixel_changes[1:], cumulative_changes[1:]), 1):
        stage_pct = (stage_change/final_change)*100 if final_change > 0 else 0
        status_icon = "🔴" if stage_change > 40 else "🟡" if stage_change > 20 else "🟢"
        analysis_text += f"   {status_icon} {name.split()[0]}: {stage_change:.1f} units ({stage_pct:.1f}%)\n"
    
    analysis_text += f"""
💡 OPTIMIZATION RECOMMENDATIONS:
   • Focus on reducing: {max_stage_name} parameters
   • Current impact level: {target_status}
   • Parameter set: {param_label}
"""
    
    ax_text.text(0.02, 0.98, analysis_text, transform=ax_text.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.6", facecolor='lightblue', alpha=0.8))
    
    # Overall title with proper spacing
    fig.suptitle(f'Virchow Augmentation Stage Analysis: {filename} ({param_label} Parameters)', 
                 fontsize=18, fontweight='bold', y=0.96)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return max_stage_name, max_stage_change, final_change


def process_stage_analysis(input_dir: str) -> None:
    """Process all images with stage-by-stage analysis for both parameter sets."""
    
    print("=" * 80)
    print("VIRCHOW COLOR JITTER STAGE-BY-STAGE ANALYSIS")
    print("=" * 80)
    print("🎯 PURPOSE: Identify which augmentation stage causes excessive pixel changes")
    print("📊 TARGET: 20-40 total intensity units for neuropathology")
    print()
    
    results_dir = create_stage_analysis_directory(input_dir)
    
    # Find image files
    input_path = Path(input_dir)
    image_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    
    if not image_files:
        print(f"❌ No image files found in {input_dir}")
        return
    
    print(f"📁 Found {len(image_files)} image(s) for stage analysis")
    print(f"🔬 Analyzing both ORIGINAL and OPTIMIZED parameter sets...")
    print()
    
    analysis_results = []
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] Analyzing: {image_file.name}")
            
            # Load original image
            original_image = load_image(str(image_file))
            print(f"  📐 Image size: {original_image.shape[1]}×{original_image.shape[0]} pixels")
            
            base_name = image_file.stem
            
            # Analyze with ORIGINAL parameters
            print("  🔬 Analyzing with ORIGINAL parameters...")
            orig_stages, orig_names, orig_changes, orig_cumulative, orig_label = \
                apply_stage_by_stage_augmentation(original_image, seed=i*42, use_optimized_params=False)
            
            orig_output_path = os.path.join(results_dir, f"{base_name}_original_stage_analysis.jpg")
            orig_max_stage, orig_max_change, orig_final = create_stage_progression_visualization(
                orig_stages, orig_names, orig_changes, orig_cumulative, 
                base_name, orig_output_path, orig_label)
            
            # Analyze with OPTIMIZED parameters  
            print("  🔬 Analyzing with OPTIMIZED parameters...")
            opt_stages, opt_names, opt_changes, opt_cumulative, opt_label = \
                apply_stage_by_stage_augmentation(original_image, seed=i*42, use_optimized_params=True)
            
            opt_output_path = os.path.join(results_dir, f"{base_name}_optimized_stage_analysis.jpg")
            opt_max_stage, opt_max_change, opt_final = create_stage_progression_visualization(
                opt_stages, opt_names, opt_changes, opt_cumulative, 
                base_name, opt_output_path, opt_label)
            
            # Store results for summary
            analysis_results.append({
                'filename': image_file.name,
                'orig_final': orig_final,
                'orig_max_stage': orig_max_stage,
                'orig_max_change': orig_max_change,
                'opt_final': opt_final,
                'opt_max_stage': opt_max_stage,
                'opt_max_change': opt_max_change,
            })
            
            print(f"  ✅ Original: {orig_final:.1f} units (max: {orig_max_stage} {orig_max_change:.1f})")
            print(f"  ✅ Optimized: {opt_final:.1f} units (max: {opt_max_stage} {opt_max_change:.1f})")
            print(f"  📁 Saved stage analyses to: {results_dir}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error analyzing {image_file.name}: {e}")
            print()
            continue
    
    # Generate summary analysis
    print("=" * 80)
    print("COMPREHENSIVE STAGE ANALYSIS SUMMARY")
    print("=" * 80)
    
    if analysis_results:
        print("📊 RESULTS COMPARISON:")
        print(f"{'Image':<15} {'Original':<12} {'Optimized':<12} {'Improvement':<12} {'Orig Max Stage':<15} {'Opt Max Stage':<15}")
        print("-" * 95)
        
        for result in analysis_results:
            improvement = ((result['orig_final'] - result['opt_final']) / result['orig_final']) * 100
            print(f"{result['filename']:<15} {result['orig_final']:<12.1f} {result['opt_final']:<12.1f} "
                  f"{improvement:<12.1f}% {result['orig_max_stage']:<15} {result['opt_max_stage']:<15}")
        
        # Calculate averages
        avg_orig = np.mean([r['orig_final'] for r in analysis_results])
        avg_opt = np.mean([r['opt_final'] for r in analysis_results])
        avg_improvement = ((avg_orig - avg_opt) / avg_orig) * 100
        
        print("-" * 95)
        print(f"{'AVERAGE':<15} {avg_orig:<12.1f} {avg_opt:<12.1f} {avg_improvement:<12.1f}%")
        
        # Identify most problematic stages
        orig_max_stages = [r['orig_max_stage'] for r in analysis_results]
        opt_max_stages = [r['opt_max_stage'] for r in analysis_results]
        
        from collections import Counter
        orig_stage_counts = Counter(orig_max_stages)
        opt_stage_counts = Counter(opt_max_stages)
        
        print(f"\n🎯 MOST PROBLEMATIC STAGES:")
        print(f"   Original parameters: {orig_stage_counts.most_common(1)[0][0]} (appears {orig_stage_counts.most_common(1)[0][1]}/{len(analysis_results)} times)")
        print(f"   Optimized parameters: {opt_stage_counts.most_common(1)[0][0]} (appears {opt_stage_counts.most_common(1)[0][1]}/{len(analysis_results)} times)")
        
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        if avg_opt > 40:
            print(f"   🔴 Further optimization needed (current: {avg_opt:.1f}, target: <40)")
            print(f"   🎯 Focus on: {opt_stage_counts.most_common(1)[0][0]} parameters")
        elif avg_opt < 20:
            print(f"   🟡 May be too conservative (current: {avg_opt:.1f}, target: >20)")
            print(f"   🎯 Consider slightly increasing parameters")
        else:
            print(f"   🟢 Within target range! (current: {avg_opt:.1f}, target: 20-40)")


def main():
    """Main execution function for stage analysis."""
    input_directory = "test_virchow_color_jitter"
    
    if not os.path.exists(input_directory):
        print(f"❌ Input directory not found: {input_directory}")
        return 1
    
    try:
        process_stage_analysis(input_directory)
        return 0
        
    except Exception as e:
        print(f"❌ Stage analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code) 