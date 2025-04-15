"""
HED Stain Augmentation Script (hed_transform.py)

This script applies the HEDTransform from the REET toolbox to simulate variations in Hematoxylin, Eosin, and DAB stain concentrations in histopathology images. It reads images from a specified input directory, applies randomized multiplicative (alpha) and additive (beta) perturbations to the H/E/D channels, and saves side-by-side comparison images (Original vs. HED-Augmented) in a results subdirectory.

Rationale:
- The HEDTransform allows direct manipulation of stain channels after color deconvolution, simulating real-world variability in staining protocols and concentrations.
- Randomizing alpha (multiplicative, around 1) and beta (additive, around 0) for each channel is a standard augmentation approach, as reflected in the REET codebase and literature.

Key Features:
- Reads images from `test_batch_HEDTransform`.
- Applies HEDTransform with random alpha in [0.8, 1.2] and beta in [-0.2, 0.2] for each channel.
- Saves side-by-side comparison images in `test_batch_HEDTransform/results`.
- Fully documented for reproducibility and clarity.

Usage:
1. Place original images in `test_batch_HEDTransform`.
2. Run: `python3 hed_transform.py`
3. Results will be in `test_batch_HEDTransform/results`.
"""

import os
import cv2
import torch
import numpy as np
import random
from REET.reetoolbox.reetoolbox.transforms import HEDTransform

def apply_hed_transform(input_dir, results_subdir_name, device):
    """Applies HEDTransform with random alpha/beta to images and saves side-by-side comparisons."""
    print(f"Input directory: {input_dir}")
    output_dir = os.path.join(input_dir, results_subdir_name)
    print(f"Output directory for results: {output_dir}")
    print(f"Using device: {device}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    image_files = [f for f in image_files if os.path.isfile(os.path.join(input_dir, f))]
    print(f"Found {len(image_files)} images to process in {input_dir}.")
    if not image_files:
        print(f"No compatible images found directly in {input_dir}. Exiting.")
        return

    # Determine input shape from the first valid image
    input_shape = None
    sample_img_loaded = False
    img_height, img_width, channels = 0, 0, 0
    for fname in image_files:
        first_image_path = os.path.join(input_dir, fname)
        try:
            sample_img = cv2.imread(first_image_path)
            if sample_img is not None:
                img_height, img_width, channels = sample_img.shape
                input_shape = (1, channels, img_height, img_width)
                print(f"Detected image shape (H, W, C): ({img_height}, {img_width}, {channels}) from {fname}")
                sample_img_loaded = True
                break
            else:
                print(f"Warning: Could not read sample image {fname}. Trying next.")
        except Exception as e:
            print(f"Warning: Error reading sample image {fname}: {e}. Trying next.")
    if not sample_img_loaded or input_shape is None:
        print("Error: Could not load any valid sample image to determine input shape. Exiting.")
        return

    # Instantiate HEDTransform
    try:
        hed_transformer = HEDTransform(input_shape=input_shape, device=device)
        print("HEDTransform instantiated.")
    except Exception as e:
        print(f"Error instantiating HEDTransform: {e}")
        print("Please ensure REET is correctly installed and accessible.")
        return

    processed_count = 0
    error_count = 0
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_filename = f"comparison_hed_{filename}"
        output_path = os.path.join(output_dir, output_filename)
        print(f"\nProcessing {filename}...")
        try:
            img_bgr = cv2.imread(input_path)
            if img_bgr is None:
                print(f"  Skipping: Could not read image {input_path}")
                error_count += 1
                continue
            if img_bgr.shape[0] != img_height or img_bgr.shape[1] != img_width or img_bgr.shape[2] != channels:
                print(f"  Skipping: Image {filename} has shape {img_bgr.shape} which differs from expected ({img_height}, {img_width}, {channels}).")
                error_count += 1
                continue
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            # Randomize alpha and beta for each channel (H, E, D)
            alpha = torch.tensor([[random.uniform(0.8, 1.2) for _ in range(3)]], device=device)
            beta = torch.tensor([[random.uniform(-0.2, 0.2) for _ in range(3)]], device=device)
            hed_transformer.weights["alpha"] = alpha
            hed_transformer.weights["beta"] = beta
            # Apply HEDTransform
            with torch.no_grad():
                hed_tensor = hed_transformer.forward(img_tensor.clone())
            # Convert back to uint8 BGR for saving
            hed_img_np = hed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            hed_img_np = np.clip(hed_img_np, 0, 255).astype(np.uint8)
            hed_img_bgr = cv2.cvtColor(hed_img_np, cv2.COLOR_RGB2BGR)
            # Create side-by-side comparison
            comparison_img = create_comparison_image(img_bgr, hed_img_bgr, "Original", "HED-Augmented")
            cv2.imwrite(output_path, comparison_img)
            print(f"  Comparison image saved to: {output_path}")
            processed_count += 1
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            error_count += 1
    print(f"\nHED transformation process completed.")
    print(f"Successfully processed: {processed_count} images.")
    print(f"Errors encountered: {error_count} images.")

def create_comparison_image(img1, img2, label1, label2):
    """Creates a side-by-side comparison image from two images with labels."""
    h, w, _ = img1.shape
    padding = 10
    text_height_offset = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 0, 0)
    font_thickness = 2
    comparison_img = np.ones((h + text_height_offset, w * 2 + padding, 3), dtype=np.uint8) * 255
    comparison_img[text_height_offset:text_height_offset+h, 0:w] = img1
    comparison_img[text_height_offset:text_height_offset+h, w + padding:w * 2 + padding] = img2
    text_size1, _ = cv2.getTextSize(label1, font, font_scale, font_thickness)
    text_x1 = (w - text_size1[0]) // 2
    text_y1 = text_height_offset - 15
    cv2.putText(comparison_img, label1, (text_x1, text_y1), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    text_size2, _ = cv2.getTextSize(label2, font, font_scale, font_thickness)
    text_x2 = w + padding + (w - text_size2[0]) // 2
    text_y2 = text_height_offset - 15
    cv2.putText(comparison_img, label2, (text_x2, text_y2), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    return comparison_img

if __name__ == "__main__":
    INPUT_IMAGE_DIR = "test_batch_HEDTransform"
    RESULTS_SUBDIR = "results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apply_hed_transform(INPUT_IMAGE_DIR, RESULTS_SUBDIR, DEVICE) 