"""
Applies Gaussian Noise and Gaussian Blur augmentations to images directly using PyTorch/Torchvision.

This script processes images from a specified input directory. For each image, it applies:
1. Global Gaussian Noise.
2. Global Gaussian Blur.
It saves side-by-side comparison images (Original vs. Augmented) for each
transformation type into a results subdirectory.

**Implementation Choice Rationale:**
This script implements standard global Gaussian Noise and Gaussian Blur directly using
PyTorch (`torch.randn_like`) and Torchvision (`torchvision.transforms.functional.gaussian_blur`)
functions. This approach was chosen because:
- The REET toolbox version available did not contain specific classes for *global*
  Gaussian Noise or *global* Gaussian Blur (`GaussianNoiseTransform`, `GaussianBlurTransform`).
- Existing REET classes like `PixelTransform` (uniform noise) and `BlurTransform`
  (regional blur) were not direct matches for the standard global augmentations desired.
- Direct implementation ensures the use of standard, well-understood methods for these
  common augmentations, providing clarity and aligning with typical machine learning practices.

Key functionalities:
- Reads images (JPG, PNG, TIF) from an input folder.
- Determines image dimensions automatically from the first valid image.
- Defines parameters for noise (mean, std) and blur (kernel size, sigma).
- Processes each image:
    - Applies global Gaussian Noise and saves an "Original vs. Noisy" comparison image.
    - Applies global Gaussian Blur and saves an "Original vs. Blurred" comparison image.
- Saves comparison images to a 'results' subdirectory within the input folder.
- Handles potential errors during file reading, processing, and directory creation.
- Uses CUDA if available, otherwise defaults to CPU.
"""

import os
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as TF # Import functional transforms from Torchvision
import random # Needed if we want randomized parameters later

# REET transforms are no longer imported as we implement noise/blur directly
# from REET.reetoolbox.reetoolbox.transforms import GaussianNoiseTransform, GaussianBlurTransform

def apply_noise_and_blur_transforms(input_dir, results_subdir_name, device):
    """Applies Gaussian Noise and Blur directly, saving comparison images.

    Reads images from `input_dir`, applies global Gaussian Noise and global
    Gaussian Blur (defined by hardcoded parameters) separately to each image
    using the specified `device` via direct PyTorch/Torchvision functions.
    Saves two comparison images per original image (Original vs. Noisy,
    Original vs. Blurred) into a subdirectory named `results_subdir_name`
    within `input_dir`.

    Args:
        input_dir (str): Path to the directory containing input images.
                         Supported formats: .png, .jpg, .jpeg, .tif, .tiff.
        results_subdir_name (str): Name of the subdirectory within `input_dir`
                                   where the comparison images will be saved.
                                   This directory will be created if it doesn't exist.
        device (torch.device): The PyTorch device ('cuda' or 'cpu') to use for
                               tensor operations.
    """
    print(f"Input directory: {input_dir}")
    output_dir = os.path.join(input_dir, results_subdir_name)
    print(f"Output directory for results: {output_dir}")
    print(f"Using device: {device}")

    # --- Input/Output Directory Handling ---
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

    # --- Discover Image Files ---
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    image_files = [f for f in image_files if os.path.isfile(os.path.join(input_dir, f))]
    print(f"Found {len(image_files)} images to process in {input_dir}.")
    if not image_files:
        print(f"No compatible images found directly in {input_dir}. Exiting.")
        return

    # --- Determine Input Shape (Required for tensor conversion, not transforms themselves now) ---
    # We still need dimensions for validation and processing
    input_shape = None # Placeholder, not strictly needed by transforms now
    sample_img_loaded = False
    img_height, img_width, channels = 0, 0, 0 # Initialize outside loop
    for fname in image_files:
        first_image_path = os.path.join(input_dir, fname)
        try:
            sample_img = cv2.imread(first_image_path)
            if sample_img is not None:
                img_height, img_width, channels = sample_img.shape
                # input_shape = (1, channels, img_height, img_width) # We don't strictly need this for direct functions
                print(f"Detected image shape (H, W, C): ({img_height}, {img_width}, {channels}) from {fname}")
                sample_img_loaded = True
                break
            else:
                 print(f"Warning: Could not read sample image {fname}. Trying next.")
        except Exception as e:
            print(f"Warning: Error reading sample image {fname}: {e}. Trying next.")
    if not sample_img_loaded:
         print("Error: Could not load any valid sample image to determine dimensions. Exiting.")
         return

    # --- Configure Transform Parameters ---
    # Gaussian Noise Parameters
    noise_mean = 0.0
    noise_std = 25.0  # Standard deviation of noise (pixel units 0-255). Adjust for more/less noise.

    # Gaussian Blur Parameters
    blur_kernel_size = 5  # Kernel size for Gaussian blur (must be odd, e.g., 3, 5, 7)
    blur_sigma = 1.5    # Standard deviation for Gaussian blur. Adjust for more/less blur.
    # Ensure kernel size is odd
    if blur_kernel_size % 2 == 0:
        print(f"Warning: Blur kernel size ({blur_kernel_size}) must be odd. Incrementing to {blur_kernel_size + 1}.")
        blur_kernel_size += 1
    # ------------------------------------

    # --- Instantiate Transforms (No longer needed) ---
    # Removed REET transform instantiation
    # print("GaussianNoiseTransform and GaussianBlurTransform instantiated.") # Removed

    # --- Initialize Counters ---
    noise_processed_count = 0
    blur_processed_count = 0
    error_count = 0

    # --- Main Processing Loop ---
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        # Define output filenames (one for noise, one for blur)
        noise_output_filename = f"comparison_noise_m{int(noise_mean)}_s{int(noise_std)}_{filename}"
        noise_output_path = os.path.join(output_dir, noise_output_filename)
        blur_output_filename = f"comparison_blur_k{blur_kernel_size}_s{blur_sigma:.1f}_{filename}"
        blur_output_path = os.path.join(output_dir, blur_output_filename)

        print(f"\nProcessing {filename}...")

        try:
            # --- Read Original Image ---
            img_original_bgr = cv2.imread(input_path)
            if img_original_bgr is None:
                print(f"  Skipping: Could not read image {input_path}")
                error_count += 1
                continue

            # --- Validate Image Dimensions ---
            if img_original_bgr.shape[0] != img_height or img_original_bgr.shape[1] != img_width or img_original_bgr.shape[2] != channels:
                 print(f"  Skipping: Image {filename} has shape {img_original_bgr.shape} which differs from expected ({img_height}, {img_width}, {channels}).")
                 error_count +=1
                 continue

            # --- Prepare Original Tensor ---
            # Convert BGR (OpenCV) to RGB
            img_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
            # Convert to PyTorch Tensor (N, C, H, W), ensure float type, move to device
            # Note: Assuming pixel values 0-255 for noise calculation
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # ===================================
            # --- Apply Gaussian Noise (Directly) ---
            # ===================================
            try:
                # Generate Gaussian noise with the same shape as the image tensor
                # noise_std is scaled relative to pixel range 0-255
                noise = torch.randn_like(img_tensor) * noise_std + noise_mean

                # Add noise to the original tensor
                noisy_tensor = img_tensor + noise

                # Clamp values to the valid range [0, 255]
                noisy_tensor = torch.clamp(noisy_tensor, 0, 255)

                # Convert back to OpenCV format (uint8 BGR)
                noisy_img_np = noisy_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                noisy_img_bgr = cv2.cvtColor(noisy_img_np, cv2.COLOR_RGB2BGR)

                # Create and Save Noise Comparison Image
                comparison_noise_img = create_comparison_image(img_original_bgr, noisy_img_bgr, "Original", f"Noisy (std={noise_std})")
                cv2.imwrite(noise_output_path, comparison_noise_img)
                print(f"  Noise comparison image saved to: {noise_output_path}")
                noise_processed_count += 1

            except Exception as e_noise:
                print(f"  Error applying Gaussian Noise to {filename}: {e_noise}")
                error_count += 1

            # ===================================
            # --- Apply Gaussian Blur (Directly) ---
            # ===================================
            try:
                # Apply Gaussian Blur using torchvision.transforms.functional
                # Input tensor should be (N, C, H, W), float type
                # We apply it to the *original* tensor, not the noisy one.
                # TF.gaussian_blur expects kernel_size to be (height, width) or a single odd int
                # Sigma can be float or tuple (sigma_y, sigma_x)
                blurred_tensor = TF.gaussian_blur(img_tensor.clone(), kernel_size=blur_kernel_size, sigma=blur_sigma)

                # Clamp values just in case, although blur shouldn't take them out of range usually
                blurred_tensor = torch.clamp(blurred_tensor, 0, 255)

                # Convert back to OpenCV format (uint8 BGR)
                blurred_img_np = blurred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                blurred_img_bgr = cv2.cvtColor(blurred_img_np, cv2.COLOR_RGB2BGR)

                # Create and Save Blur Comparison Image
                comparison_blur_img = create_comparison_image(img_original_bgr, blurred_img_bgr, "Original", f"Blurred (k={blur_kernel_size}, s={blur_sigma})")
                cv2.imwrite(blur_output_path, comparison_blur_img)
                print(f"  Blur comparison image saved to: {blur_output_path}")
                blur_processed_count += 1

            except Exception as e_blur:
                print(f"  Error applying Gaussian Blur to {filename}: {e_blur}")
                error_count += 1

        except Exception as e_main:
            # Catch errors in reading, validation, or tensor prep
            print(f"  Error processing {filename} (before transforms): {e_main}")
            error_count += 1

    # --- Final Summary ---
    print(f"\nNoise and Blur transformation process completed.")
    print(f"Successfully generated: {noise_processed_count} Noise comparison images.")
    print(f"Successfully generated: {blur_processed_count} Blur comparison images.")
    print(f"Errors encountered: {error_count} images/transforms.")


def create_comparison_image(img1, img2, label1, label2):
    """Creates a side-by-side comparison image from two images with labels."""
    h, w, _ = img1.shape # Assuming img1 and img2 have the same shape
    padding = 10
    text_height_offset = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (0, 0, 0) # Black
    font_thickness = 2

    # Create canvas
    comparison_img = np.ones((h + text_height_offset, w * 2 + padding, 3), dtype=np.uint8) * 255

    # Place images
    comparison_img[text_height_offset:text_height_offset+h, 0:w] = img1
    comparison_img[text_height_offset:text_height_offset+h, w + padding:w * 2 + padding] = img2

    # Add Label 1
    text_size1, _ = cv2.getTextSize(label1, font, font_scale, font_thickness)
    text_x1 = (w - text_size1[0]) // 2
    text_y1 = text_height_offset - 15
    cv2.putText(comparison_img, label1, (text_x1, text_y1), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Add Label 2
    text_size2, _ = cv2.getTextSize(label2, font, font_scale, font_thickness)
    text_x2 = w + padding + (w - text_size2[0]) // 2
    text_y2 = text_height_offset - 15
    cv2.putText(comparison_img, label2, (text_x2, text_y2), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    return comparison_img


# --- Script Entry Point ---
if __name__ == "__main__":
    # --- Configuration --- #
    INPUT_IMAGE_DIR = "test_batch_Noise&Blur"  # Set to the correct directory
    RESULTS_SUBDIR = "results"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------- #

    # Call the main processing function
    apply_noise_and_blur_transforms(INPUT_IMAGE_DIR, RESULTS_SUBDIR, DEVICE) 