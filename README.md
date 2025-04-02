# Digital Pathology Research Tools

This repository contains tools and scripts for digital pathology research, focusing on image processing and analysis techniques for histopathology images.

## Stain Normalization

The `stain_norm.py` script implements Reinhard color normalization for histopathology images. This technique standardizes the color appearance across different images by matching their color statistics in the LAB color space.

### Features

- **Reinhard Normalization**: Implements the Reinhard method for color normalization
- **Smart Reference Selection**: Automatically selects the best reference image based on quality metrics
- **Quality Metrics**: Analyzes images based on contrast, color balance, brightness, and saturation
- **Visualization**: Generates side-by-side comparisons of original and normalized images

### Usage

1. Place your histopathology images in the `test_batch` directory
2. Run the script: `python stain_norm.py`
3. Results will be saved in the `test_batch/smart_selection_results` directory

### Requirements

- Python 3.6+
- NumPy
- OpenCV
- PIL (Pillow)
- Matplotlib

## License

[MIT License](LICENSE)

## Author

Ryan Rodriguez 