"""
Applies a regional Gaussian blur to images using the REET toolbox's BlurTransform.

This script processes images from a specified input directory, applies a
configurable Gaussian blur to a defined rectangular region within each image,
and saves a side-by-side comparison image (Original vs. Blurred) to a
results subdirectory.

**Current Implementation Note:**
This version supports both fixed and randomized regions for blurring. When using
randomized mode, the blur region (position, size) is randomly determined for each image,
providing better data augmentation for training robust machine learning models.

Key functionalities:
- Reads images (JPG, PNG, TIF) from an input folder.
- Determines image dimensions automatically from the first valid image.
- Instantiates and configures REET's BlurTransform.
- Supports both fixed and randomized blur regions:
  - Fixed mode: Uses specified parameters for all images
  - Random mode: Randomizes position and optionally size/intensity for each image
- Processes each image:
    - Applies the blur transform to the specified region.
    - Creates a composite image showing the original and blurred versions
      side-by-side with labels.
- Saves the comparison image to a 'results' subdirectory within the input folder.
- Handles potential errors during file reading, processing, and directory creation.
- Uses CUDA if available, otherwise defaults to CPU.
"""

import os
import cv2
import torch
import numpy as np
import random
import shutil # Used for file operations if needed, though currently only used for copying in commented-out code.
# Ensure the path to your REET installation is correct or that it's in your PYTHONPATH
# The script requires the REET toolbox to be installed or accessible.
# We installed it in editable mode previously using 'pip install -e .' in the REET/reetoolbox directory.
from REET.reetoolbox.reetoolbox.transforms import BlurTransform
# from REET.reetoolbox.reetoolbox.utils import utils # utils might not be strictly necessary here unless converting tensor formats


# Note: Changed output_dir logic to be relative to input_dir
def apply_blur_transform(input_dir, results_subdir_name, device, 
                         randomize_blur=True, 
                         min_region_size=50, 
                         max_region_size=200,
                         min_kernel_size=3, 
                         max_kernel_size=9,
                         min_sigma=5.0, 
                         max_sigma=15.0,
                         fixed_params=None):
    """Applies regional blur and generates side-by-side comparison images.

    Reads images from `input_dir`, applies a Gaussian blur to each image
    using the specified `device`, and saves a comparison image showing 
    the original and blurred versions side-by-side into a subdirectory
    named `results_subdir_name` within `input_dir`.

    Args:
        input_dir (str): Path to the directory containing input images.
                         Supported formats: .png, .jpg, .jpeg, .tif, .tiff.
        results_subdir_name (str): Name of the subdirectory within `input_dir`
                                   where the comparison images will be saved.
                                   This directory will be created if it doesn't exist.
        device (torch.device): The PyTorch device ('cuda' or 'cpu') to use for
                               tensor operations and the blur transformation.
        randomize_blur (bool): If True, randomizes the blur region for each image.
                              If False, uses fixed parameters for all images.
        min_region_size (int): Minimum size (height/width) for random blur regions.
        max_region_size (int): Maximum size (height/width) for random blur regions.
        min_kernel_size (int): Minimum kernel size for Gaussian blur (must be odd).
        max_kernel_size (int): Maximum kernel size for Gaussian blur (must be odd).
        min_sigma (float): Minimum sigma value for Gaussian blur.
        max_sigma (float): Maximum sigma value for Gaussian blur.
        fixed_params (dict, optional): Dictionary of fixed blur parameters to use if
                                      randomize_blur is False. If None, default values are used.
    """
    print(f"Input directory: {input_dir}")
    # Construct the full path for the output directory
    output_dir = os.path.join(input_dir, results_subdir_name)
    print(f"Output directory for results: {output_dir}")
    print(f"Using device: {device}")
    print(f"Blur mode: {'Randomized' if randomize_blur else 'Fixed'}")

    # --- Input Directory Validation ---
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        return

    # --- Output Directory Creation ---
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return

    # --- Discover Image Files ---
    # List files in the input directory and filter for common image extensions.
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    # Filter out any potential subdirectories (like the results directory itself)
    # This ensures we only process files directly within input_dir.
    image_files = [f for f in image_files if os.path.isfile(os.path.join(input_dir, f))]
    print(f"Found {len(image_files)} images to process in {input_dir}.")


    if not image_files:
        print(f"No compatible images found directly in {input_dir}. Exiting.")
        return

    # --- Determine Input Shape ---
    # REET transforms often require the expected input tensor shape (N, C, H, W).
    # We read the first valid image to determine Height, Width, and Channels.
    # Batch size (N) is assumed to be 1 as we process images individually.
    input_shape = None
    sample_img_loaded = False
    sample_img_height = None
    sample_img_width = None
    for fname in image_files:
        first_image_path = os.path.join(input_dir, fname)
        try:
            # Read image using OpenCV
            sample_img = cv2.imread(first_image_path)
            if sample_img is not None:
                # Get dimensions (OpenCV shape: H, W, C)
                img_height, img_width, channels = sample_img.shape
                sample_img_height, sample_img_width = img_height, img_width
                # Define the shape for PyTorch (N, C, H, W)
                input_shape = (1, channels, img_height, img_width)
                print(f"Detected image shape (H, W, C): ({img_height}, {img_width}, {channels}) from {fname}")
                print(f"Setting input shape for transform (N, C, H, W): {input_shape}")
                sample_img_loaded = True
                break # Successfully determined shape, exit loop
            else:
                 print(f"Warning: Could not read sample image {fname}. Trying next.")
        except Exception as e:
            print(f"Warning: Error reading sample image {fname}: {e}. Trying next.")

    # If no image could be read to determine shape, we cannot proceed.
    if not sample_img_loaded or input_shape is None:
         print("Error: Could not load any valid sample image to determine input shape. Exiting.")
         return
    # --------------------------------------------------------

    # --- Configure Default Blur Parameters --- #
    # These parameters are used for fixed mode or as fallback
    if fixed_params is None:
        fixed_params = {
            "blur_corner_x": 50,      # X-coordinate (from left) of the top-left corner of the blur region.
            "blur_corner_y": 50,      # Y-coordinate (from top) of the top-left corner of the blur region.
            "blur_height": 100,       # Height of the rectangular blur region.
            "blur_width": 100,        # Width of the rectangular blur region.
            "blur_kernel_size": 5,    # Size of the Gaussian kernel (must be odd). Larger values mean more blur.
            "blur_sigma": 10.0        # Standard deviation of the Gaussian kernel. Larger values mean more blur.
        }
    # ----------------------------------------- #

    # --- Instantiate BlurTransform ---
    try:
        # Create an instance of the BlurTransform.
        # It needs the expected input shape and the device to run on.
        blur_transformer = BlurTransform(input_shape=input_shape, device=device)
        print("BlurTransform instantiated.")
    except Exception as e:
        print(f"Error instantiating BlurTransform: {e}")
        print("Please ensure REET is correctly installed and accessible.")
        return

    # --- Initialize Counters ---
    processed_count = 0 # Counts successfully generated comparison images.
    error_count = 0     # Counts images that failed during processing.

    # --- Main Processing Loop ---
    # Iterate through each discovered image file.
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        
        print(f"\nProcessing {filename}...")

        try:
            # --- Read Original Image ---
            # Read the image using OpenCV (returns NumPy array in BGR format).
            img_original_bgr = cv2.imread(input_path)
            # Handle cases where the image cannot be read.
            if img_original_bgr is None:
                print(f"  Skipping: Could not read image {input_path}")
                error_count += 1
                continue

            # --- Validate Image Dimensions ---
            # Ensure the current image's dimensions match the shape determined earlier.
            # This prevents errors if the input directory contains images of varying sizes.
            if img_original_bgr.shape[0] != input_shape[2] or img_original_bgr.shape[1] != input_shape[3] or img_original_bgr.shape[2] != input_shape[1]:
                 print(f"  Skipping: Image {filename} has shape {img_original_bgr.shape} which differs from expected input shape ({input_shape[2]}, {input_shape[3]}, {input_shape[1]}).")
                 error_count +=1
                 continue
            
            # --- Generate Blur Parameters ---
            # Either use randomized parameters or fixed parameters based on configuration
            if randomize_blur:
                # Randomize blur region parameters
                # Ensure kernel size is odd (required for Gaussian blur)
                possible_kernel_sizes = [k for k in range(min_kernel_size, max_kernel_size + 1) if k % 2 == 1]
                if not possible_kernel_sizes:
                    # Fallback if no valid kernel sizes in range
                    possible_kernel_sizes = [min_kernel_size if min_kernel_size % 2 == 1 else min_kernel_size + 1]
                
                # Randomly select region size
                blur_height = random.randint(min_region_size, max_region_size)
                blur_width = random.randint(min_region_size, max_region_size)
                
                # Ensure region fits within image bounds
                max_corner_x = max(0, sample_img_width - blur_width - 1)
                max_corner_y = max(0, sample_img_height - blur_height - 1)
                
                # Randomly select region position
                blur_corner_x = random.randint(0, max_corner_x)
                blur_corner_y = random.randint(0, max_corner_y)
                
                # Randomly select blur parameters
                blur_kernel_size = random.choice(possible_kernel_sizes)
                blur_sigma = random.uniform(min_sigma, max_sigma)
                
                print(f"  Randomized blur parameters: position=({blur_corner_x}, {blur_corner_y}), "
                      f"size=({blur_width}, {blur_height}), kernel={blur_kernel_size}, sigma={blur_sigma:.2f}")
            else:
                # Use fixed parameters
                blur_corner_x = fixed_params["blur_corner_x"]
                blur_corner_y = fixed_params["blur_corner_y"]
                blur_height = fixed_params["blur_height"]
                blur_width = fixed_params["blur_width"]
                blur_kernel_size = fixed_params["blur_kernel_size"]
                blur_sigma = fixed_params["blur_sigma"]
                
                print(f"  Using fixed blur parameters: position=({blur_corner_x}, {blur_corner_y}), "
                      f"size=({blur_width}, {blur_height}), kernel={blur_kernel_size}, sigma={blur_sigma:.2f}")

            # --- Apply Blur Transform ---
            # 1. Convert BGR (OpenCV) to RGB (standard for many libraries including REET)
            img_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)

            # 2. Convert NumPy array (H, W, C) to PyTorch Tensor (N, C, H, W)
            #    - permute changes order of dimensions.
            #    - unsqueeze adds the batch dimension (N=1).
            #    - float converts to float type, often required by models/transforms.
            #    - to(device) moves the tensor to the designated CPU/GPU.
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # 3. Set Blur Parameters for this specific application.
            #    REET transforms often use a 'weights' dictionary to dynamically set parameters.
            #    Parameters are converted to float tensors on the correct device.
            blur_transformer.weights = {
                "corner_x": torch.tensor([[float(blur_corner_x)]], device=device),
                "corner_y": torch.tensor([[float(blur_corner_y)]], device=device),
                "height": torch.tensor([[float(blur_height)]], device=device),
                "width": torch.tensor([[float(blur_width)]], device=device),
                "kernel_size": torch.tensor([[float(blur_kernel_size)]], device=device),
                "sigma": torch.tensor([[float(blur_sigma)]], device=device)
            }

            # 4. Apply the transform.
            #    torch.no_grad() disables gradient calculation, saving memory during inference.
            #    .clone() ensures the original tensor isn't modified if the transform works in-place.
            with torch.no_grad():
                blurred_tensor = blur_transformer.forward(img_tensor.clone())

            # 5. Convert the resulting blurred tensor back to a NumPy array for OpenCV.
            #    - squeeze(0) removes the batch dimension.
            #    - permute(1, 2, 0) changes order back to (H, W, C).
            #    - cpu() moves tensor back to CPU if it was on GPU.
            #    - numpy() converts to NumPy array.
            blurred_img_np = blurred_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Ensure pixel values are within the valid range [0, 255] and correct type.
            blurred_img_np = np.clip(blurred_img_np, 0, 255).astype(np.uint8)

            # 6. Convert RGB back to BGR for saving/display with OpenCV.
            blurred_img_bgr = cv2.cvtColor(blurred_img_np, cv2.COLOR_RGB2BGR)
            # --- End Blur Transform Application ---

            # --- Create Side-by-Side Comparison Image ---
            h, w, _ = img_original_bgr.shape # Get dimensions from original
            padding = 10             # Space between the two images.
            text_height_offset = 40  # Top margin for text labels.
            font = cv2.FONT_HERSHEY_SIMPLEX # Font for labels.
            font_scale = 0.8         # Font size.
            font_color = (0, 0, 0)   # Black text.
            font_thickness = 2       # Text line thickness.

            # Create a new blank canvas (white background).
            # Width = Original Width * 2 + Padding
            # Height = Original Height + Text Offset
            comparison_img = np.ones((h + text_height_offset, w * 2 + padding, 3), dtype=np.uint8) * 255

            # Place original image onto the left side of the canvas.
            # Slicing defines the target region on the canvas.
            comparison_img[text_height_offset:text_height_offset+h, 0:w] = img_original_bgr

            # Place blurred image onto the right side of the canvas, after the padding.
            comparison_img[text_height_offset:text_height_offset+h, w + padding:w * 2 + padding] = blurred_img_bgr

            # Add "Original" Label
            text_orig = "Original"
            # Get text size to center it horizontally.
            text_size_orig, _ = cv2.getTextSize(text_orig, font, font_scale, font_thickness)
            # Calculate X position for centered text above the left image.
            text_x_orig = (w - text_size_orig[0]) // 2
            # Calculate Y position (within the top margin).
            text_y_orig = text_height_offset - 15
            # Draw the text onto the canvas.
            cv2.putText(comparison_img, text_orig, (text_x_orig, text_y_orig), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # Add "Blurred" Label with parameters
            # Create a more detailed label for the blurred image
            if randomize_blur:
                text_blurred = f"Blurred (Randomized)"
            else:
                text_blurred = f"Blurred (Fixed)"
                
            text_size_blurred, _ = cv2.getTextSize(text_blurred, font, font_scale, font_thickness)
            # Calculate X position for centered text above the right image.
            text_x_blurred = w + padding + (w - text_size_blurred[0]) // 2
            text_y_blurred = text_height_offset - 15
            cv2.putText(comparison_img, text_blurred, (text_x_blurred, text_y_blurred), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            # --------------------------------------------

            # Define the output filename for the side-by-side comparison image.
            # Include method and blur parameters in the filename.
            if randomize_blur:
                comparison_output_filename = f"comparison_random_cx{blur_corner_x}_cy{blur_corner_y}_h{blur_height}_w{blur_width}_k{blur_kernel_size}_s{int(blur_sigma)}_{filename}"
            else:
                comparison_output_filename = f"comparison_fixed_cx{blur_corner_x}_cy{blur_corner_y}_h{blur_height}_w{blur_width}_k{blur_kernel_size}_s{int(blur_sigma)}_{filename}"
                
            comparison_output_path = os.path.join(output_dir, comparison_output_filename)

            # --- Save the Comparison Image ---
            # Save the combined image to the results directory.
            cv2.imwrite(comparison_output_path, comparison_img)
            print(f"  Comparison image saved to: {comparison_output_path}")
            processed_count += 1 # Increment success counter.

        except Exception as e:
            # --- Error Handling ---
            # Catch any exceptions during the processing of a single file.
            print(f"  Error processing {filename}: {e}")
            error_count += 1 # Increment error counter.

    # --- Final Summary ---
    print(f"\nBlur transformation and comparison image generation completed.")
    print(f"Successfully generated: {processed_count} comparison images.")
    print(f"Errors encountered: {error_count} images.")

# --- Script Entry Point ---
if __name__ == "__main__":
    # This block executes only when the script is run directly (not imported).
    # --- Configuration --- #
    # Define the main input directory containing the original images.
    INPUT_IMAGE_DIR = "test_batch_BlurTransform"
    # Define the name for the subdirectory where results will be saved.
    RESULTS_SUBDIR = "results_random"
    # Automatically select CUDA (GPU) if available, otherwise use CPU.
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration for blur randomization
    RANDOMIZE_BLUR = True  # Set to False to use fixed parameters
    
    # Ranges for randomization (used only when RANDOMIZE_BLUR is True)
    MIN_REGION_SIZE = 50   # Minimum height/width of blur region
    MAX_REGION_SIZE = 200  # Maximum height/width of blur region
    MIN_KERNEL_SIZE = 3    # Minimum blur kernel size (must be odd)
    MAX_KERNEL_SIZE = 9    # Maximum blur kernel size (must be odd)
    MIN_SIGMA = 5.0        # Minimum blur sigma (standard deviation)
    MAX_SIGMA = 15.0       # Maximum blur sigma (standard deviation)
    # ------------------- #

    # Call the main processing function with the configured parameters.
    apply_blur_transform(
        input_dir=INPUT_IMAGE_DIR,
        results_subdir_name=RESULTS_SUBDIR,
        device=DEVICE,
        randomize_blur=RANDOMIZE_BLUR,
        min_region_size=MIN_REGION_SIZE,
        max_region_size=MAX_REGION_SIZE,
        min_kernel_size=MIN_KERNEL_SIZE,
        max_kernel_size=MAX_KERNEL_SIZE,
        min_sigma=MIN_SIGMA,
        max_sigma=MAX_SIGMA
    )
