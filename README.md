# Digital Pathology Research Tools

This repository contains tools and scripts for digital pathology research, focusing on image processing and analysis techniques for histopathology images.

## Python Scripts

### Stain Normalization (`stain_norm.py`)

This script implements Reinhard color normalization for histopathology images. This technique standardizes the color appearance across different images by matching their color statistics in the LAB color space.

**Features:**

- **Reinhard Normalization**: Implements the Reinhard method for color normalization.
- **Smart Reference Selection**: Automatically selects the best reference image based on quality metrics (contrast, color balance, brightness, saturation).
- **Quality Metrics**: Analyzes images based on contrast, color balance, brightness, and saturation.
- **Visualization**: Generates side-by-side comparisons of original and normalized images.

**Usage:**

1. Place your histopathology images in the `test_batch_Reinhard` directory.
2. Run the script: `python3 stain_norm.py`
3. Results (comparison images) will be saved in the `test_batch_Reinhard/results` directory.

### Regional Blur Transformation (`blur_transform.py`)

This script applies a Gaussian blur to a specific *region* within histopathology images using the `BlurTransform` class from the REET toolbox.

**Features:**

- **Regional Blur**: Applies Gaussian blur to a defined rectangular area.
- **Configurable Parameters**: Blur intensity (kernel size, sigma) and region (corner coordinates, height, width) can be configured within the script.
- **Visualization**: Generates side-by-side comparisons of the original image and the image with the regional blur applied.

**Current Implementation Note:**
This script currently uses a **fixed, hardcoded region** for blurring (100x100 square starting at 50,50). This was implemented for simplicity during initial setup. For effective data augmentation, future enhancements could involve **randomizing** the blur region's location and potentially its size/intensity for each image processed.

**Usage:**

1. Place your histopathology images in the `test_batch_BlurTransform` directory.
2. Ensure the REET library is installed in your environment (e.g., `python3 -m pip install -e REET/reetoolbox/` from the project root).
3. Run the script: `python3 blur_transform.py`
4. Results (comparison images with regional blur) will be saved in the `test_batch_BlurTransform/results` directory.

### Global Noise & Blur Transformations (`noise_blur_transforms.py`)

This script applies standard *global* Gaussian Noise and *global* Gaussian Blur augmentations to images directly using PyTorch and Torchvision functions.

**Implementation Rationale:**
This direct implementation approach was chosen because the available REET toolbox version lacked specific classes for global Gaussian noise/blur, and adapting existing regional/uniform REET transforms was less suitable than using standard, widely-used PyTorch/Torchvision functions for these common augmentations.

**Features:**

- **Global Gaussian Noise**: Adds noise following a normal distribution over the entire image.
- **Global Gaussian Blur**: Applies Gaussian blur across the entire image.
- **Configurable Parameters**: Noise level (mean, standard deviation) and blur intensity (kernel size, sigma) can be configured within the script.
- **Visualization**: Generates *two* separate side-by-side comparison images for each original: one showing "Original vs. Noisy" and another showing "Original vs. Blurred".

**Usage:**

1. Place your histopathology images in the `test_batch_Noise&Blur` directory.
2. Run the script: `python3 noise_blur_transforms.py`
3. Results (separate comparison images for noise and blur effects) will be saved in the `test_batch_Noise&Blur/results` directory.

## Test Data Directories

These directories contain sample images used for testing the different augmentation scripts. Each directory typically contains original images and a `results` subdirectory where the output comparison images are saved.

- **`test_batch_Reinhard/`**: Input images for testing `stain_norm.py`. Results are saved in `test_batch_Reinhard/results/`.
- **`test_batch_BlurTransform/`**: Input images for testing `blur_transform.py` (regional blur). Results are saved in `test_batch_BlurTransform/results/`.
- **`test_batch_Noise&Blur/`**: Input images for testing `noise_blur_transforms.py` (global noise and blur). Results are saved in `test_batch_Noise&Blur/results/`.
- **`Test_Images/`**: (If still present) May contain earlier test images or originals. Currently not directly used by the primary scripts.

## Requirements

- Python 3.6+
- NumPy
- OpenCV (`opencv-python`)
- PIL (Pillow)
- Matplotlib
- PyTorch (`torch`)
- Torchvision (`torchvision`)
- REET Toolbox (required for `blur_transform.py`, install via `pip install -e REET/reetoolbox/`)

## License

[MIT License](LICENSE)

## Author

Ryan Rodriguez 