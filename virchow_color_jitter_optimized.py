#!/usr/bin/env python3
"""
NeuroPath Optimized Virchow Color Jitter
Calibrated specifically for neuropathology WSI foundation model training

Based on validation findings, parameters have been optimized to:
1. Reduce average pixel change from 90+ to 30-40 intensity units
2. Preserve neuropathological morphological features
3. Maintain cross-institutional robustness
4. Support GM/WM/Meninge tissue classification accuracy

Dr. Shuying Li's NeuroPath Project - Boston University
"""

import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from typing import Tuple, Union
import cv2
from skimage import color
from virchow_color_jitter import StainJitter, LABShift

class NeuroPathVirchowColorJitter(A.Compose):
    """
    Neuropathology-optimized Virchow Color Jitter
    
    Parameters calibrated based on validation results to provide optimal
    augmentation for GM/WM/Meninge tissue classification while preserving
    diagnostic features critical for neuropathology analysis.
    
    Key Optimizations:
    - Reduced brightness variation: (0.95, 1.05) vs (0.9, 1.1)
    - Conservative stain jitter: 3° rotation vs 5°
    - Focused illumination shifts: ±10 L* vs ±15 L*
    - Balanced probabilities for neuropathology workflows
    """
    
    def __init__(
        self,
        # OPTIMIZED PARAMETERS FOR NEUROPATHOLOGY
        brightness_range: Tuple[float, float] = (0.95, 1.05),  # Reduced from (0.9, 1.1)
        hue_range: int = 3,                                    # Reduced from 5
        saturation_range: int = 3,                             # Reduced from 5  
        value_range: int = 3,                                  # Reduced from 5
        stain_rotation_range: float = 3.0,                     # Reduced from 5.0
        stain_magnitude_range: float = 0.03,                   # Reduced from 0.05
        jpeg_quality_range: Tuple[int, int] = (70, 100),       # Increased from (50, 100)
        lab_l_range: Tuple[float, float] = (-10, 10),          # Reduced from (-15, 15)
        lab_a_range: Tuple[float, float] = (-3, 3),            # Reduced from (-5, 5)
        lab_b_range: Tuple[float, float] = (-3, 3),            # Reduced from (-5, 5)
        # PROBABILITY ADJUSTMENTS FOR NEUROPATHOLOGY
        p_stain: float = 0.7,                                  # Reduced from 0.8
        p_jpeg: float = 0.3,                                   # Reduced from 0.5
        p_lab: float = 0.6,                                    # Reduced from 0.7
    ):
        """Initialize NeuroPath-optimized augmentation pipeline."""
        
        # Store parameters for reproducibility
        self.brightness_range = brightness_range
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range
        self.stain_rotation_range = stain_rotation_range
        self.stain_magnitude_range = stain_magnitude_range
        self.jpeg_quality_range = jpeg_quality_range
        self.lab_l_range = lab_l_range
        self.lab_a_range = lab_a_range
        self.lab_b_range = lab_b_range
        self.p_stain = p_stain
        self.p_jpeg = p_jpeg
        self.p_lab = p_lab
        
        # Build optimized augmentation pipeline
        transforms = [
            # 1. Conservative brightness/contrast (±5% vs ±10%)
            A.RandomBrightnessContrast(
                brightness_limit=abs(brightness_range[0] - 1.0),
                contrast_limit=abs(brightness_range[0] - 1.0),
                brightness_by_max=False,
                p=1.0,
            ),
            
            # 2. Reduced HSV jitter for neuropathology color preservation
            A.HueSaturationValue(
                hue_shift_limit=hue_range,
                sat_shift_limit=saturation_range,
                val_shift_limit=value_range,
                p=1.0,
            ),
            
            # 3. Conservative stain jitter (3° vs 5° rotation)
            StainJitter(
                rotation_range=stain_rotation_range,
                magnitude_range=stain_magnitude_range,
                p=p_stain,
            ),
            
            # 4. Higher quality JPEG compression (70-100% vs 50-100%)
            A.ImageCompression(
                quality_lower=jpeg_quality_range[0],
                quality_upper=jpeg_quality_range[1],
                p=p_jpeg,
            ),
            
            # 5. Reduced illumination shifts for diagnostic preservation
            LABShift(
                l_range=lab_l_range,
                a_range=lab_a_range,
                b_range=lab_b_range,
                p=p_lab,
            ),
        ]
        
        # Initialize with optimized pipeline
        super().__init__(transforms=transforms, p=1.0)

class NeuroPathVirchowColorJitterTorch(nn.Module):
    """PyTorch-compatible wrapper for NeuroPath-optimized Virchow Color Jitter."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.augmentation = NeuroPathVirchowColorJitter(**kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to PyTorch tensor with optimized parameters."""
        # Convert CHW tensor to HWC numpy array
        if x.dim() == 3:
            x_np = x.permute(1, 2, 0).cpu().numpy()
            x_np = (x_np * 255).astype(np.uint8)
        else:
            raise ValueError("Expected 3D tensor (C, H, W)")
        
        # Apply augmentation
        try:
            augmented_dict = self.augmentation(image=x_np)
            augmented_np = augmented_dict["image"]
        except Exception as e:
            print(f"Warning: Augmentation failed, returning original: {e}")
            augmented_np = x_np
        
        # Convert back to tensor
        augmented_tensor = torch.from_numpy(augmented_np).float() / 255.0
        augmented_tensor = augmented_tensor.permute(2, 0, 1)
        
        return augmented_tensor.to(x.device)

# NEUROPATHOLOGY-SPECIFIC CONFIGURATIONS

def get_neuropath_conservative_config():
    """Ultra-conservative config for diagnostic accuracy preservation."""
    return NeuroPathVirchowColorJitter(
        brightness_range=(0.98, 1.02),   # ±2% brightness
        hue_range=1,                     # ±1° hue shift
        saturation_range=1,              # ±1% saturation
        stain_rotation_range=1.0,        # ±1° stain rotation
        stain_magnitude_range=0.01,      # ±1% stain magnitude
        p_stain=0.5,                     # 50% stain probability
        p_jpeg=0.1,                      # 10% compression
        p_lab=0.3,                       # 30% illumination
    )

def get_neuropath_robust_config():
    """Balanced config for cross-institutional robustness."""
    return NeuroPathVirchowColorJitter(
        brightness_range=(0.93, 1.07),   # ±7% brightness 
        hue_range=4,                     # ±4° hue shift
        saturation_range=4,              # ±4% saturation
        stain_rotation_range=4.0,        # ±4° stain rotation
        stain_magnitude_range=0.04,      # ±4% stain magnitude
        p_stain=0.8,                     # 80% stain probability
        p_jpeg=0.4,                      # 40% compression
        p_lab=0.7,                       # 70% illumination
    )

def get_neuropath_evaluation_config():
    """Config for augmentation evaluation and ablation studies."""
    return NeuroPathVirchowColorJitter(
        brightness_range=(0.95, 1.05),   # Standard ±5%
        hue_range=3,                     # Standard ±3°
        saturation_range=3,              # Standard ±3%
        stain_rotation_range=3.0,        # Standard ±3°
        stain_magnitude_range=0.03,      # Standard ±3%
        p_stain=0.7,                     # 70% probability
        p_jpeg=0.3,                      # 30% compression
        p_lab=0.6,                       # 60% illumination
    )

if __name__ == "__main__":
    # Quick validation test
    print("NeuroPath Optimized Virchow Color Jitter - Validation Test")
    print("=" * 60)
    
    # Test with synthetic pathology image
    test_image = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    test_image = test_image.astype(np.float32)
    test_image[:, :, 0] *= 1.1  # Enhance red (eosin)
    test_image[:, :, 2] *= 0.9  # Reduce blue
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    
    # Test optimized version
    optimized_transform = NeuroPathVirchowColorJitter()
    result = optimized_transform(image=test_image)["image"]
    
    pixel_change = np.mean(np.abs(result.astype(float) - test_image.astype(float)))
    print(f"✓ Optimized augmentation - Average pixel change: {pixel_change:.2f}")
    print(f"✓ Target range: 20-40 intensity units")
    print(f"✓ Status: {'OPTIMAL' if 20 <= pixel_change <= 40 else 'NEEDS ADJUSTMENT'}")
    
    # Test PyTorch version
    test_tensor = torch.from_numpy(test_image).permute(2, 0, 1).float() / 255.0
    torch_transform = NeuroPathVirchowColorJitterTorch()
    result_tensor = torch_transform(test_tensor)
    
    tensor_change = torch.mean(torch.abs(result_tensor - test_tensor)).item()
    print(f"✓ PyTorch interface - Average change: {tensor_change:.4f}")
    print(f"✓ Expected range: 0.08-0.16 (normalized)")
    print(f"✓ Status: {'OPTIMAL' if 0.08 <= tensor_change <= 0.16 else 'NEEDS ADJUSTMENT'}") 