import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

class ReinhardNormalizer:
    """
    Implements Reinhard color normalization for histopathology images.
    
    This class provides methods to normalize histopathology images using the Reinhard method,
    which standardizes the color appearance across different images by matching their color
    statistics in the LAB color space.
    
    Attributes:
        target_means (numpy.ndarray): Mean values of the reference image in LAB color space
        target_stds (numpy.ndarray): Standard deviation values of the reference image in LAB color space
    """
    
    def __init__(self):
        """
        Initialize the ReinhardNormalizer with empty target statistics.
        """
        self.target_means = None
        self.target_stds = None
    
    def fit(self, target_image):
        """
        Compute the target statistics from a reference image.
        
        This method calculates the mean and standard deviation of the reference image
        in the LAB color space, which will be used as the target for normalizing other images.
        
        Args:
            target_image (numpy.ndarray): Reference image in RGB format (height, width, 3)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB)
        
        # Calculate mean and std for each channel
        self.target_means = np.mean(lab, axis=(0, 1))
        self.target_stds = np.std(lab, axis=(0, 1))
    
    def normalize(self, image):
        """
        Normalize an image using Reinhard normalization.
        
        This method applies the Reinhard normalization to the input image by matching
        its color statistics to those of the reference image in the LAB color space.
        
        Args:
            image (numpy.ndarray): Input image in RGB format (height, width, 3)
            
        Returns:
            numpy.ndarray: Normalized image in RGB format (height, width, 3)
            
        Raises:
            ValueError: If the normalizer hasn't been fit with a target image
        """
        if self.target_means is None:
            raise ValueError("Normalizer must be fit with a target image first")
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Calculate source statistics
        src_means = np.mean(lab, axis=(0, 1))
        src_stds = np.std(lab, axis=(0, 1))
        
        # Normalize each channel
        normalized_lab = np.copy(lab).astype(np.float32)
        for i in range(3):
            normalized_lab[:,:,i] = ((normalized_lab[:,:,i] - src_means[i]) * 
                                   (self.target_stds[i] / src_stds[i])) + self.target_means[i]
        
        # Clip values to valid range
        normalized_lab = np.clip(normalized_lab, 0, 255)
        
        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return normalized_rgb

def load_image(image_path):
    """
    Load an image and convert to RGB format.
    
    This function opens an image file and converts it to RGB format,
    ensuring consistent color representation across different image formats.
    
    Args:
        image_path (Path or str): Path to the image file
        
    Returns:
        numpy.ndarray: Image in RGB format (height, width, 3)
    """
    return np.array(Image.open(image_path).convert("RGB"))

def analyze_image_quality(image):
    """
    Analyze image quality metrics.
    
    This function calculates various quality metrics for an image in the LAB color space:
    - Contrast: Variation in brightness (using L channel standard deviation)
    - Color Balance: Ratio of a and b channel variations
    - Brightness: Overall brightness level (L channel mean)
    - Saturation: Color intensity (average of a and b channel stds)
    - Overall Score: Combined score based on contrast and saturation
    
    Args:
        image (numpy.ndarray): Input image in RGB format (height, width, 3)
        
    Returns:
        dict: Dictionary containing quality metrics
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Calculate basic statistics
    means = np.mean(lab, axis=(0, 1))
    stds = np.std(lab, axis=(0, 1))
    
    # Calculate contrast (using standard deviation of L channel)
    contrast = stds[0]
    
    # Calculate color balance (ratio of a and b channel stds)
    color_balance = stds[1] / stds[2] if stds[2] != 0 else float('inf')
    
    # Calculate brightness (L channel mean)
    brightness = means[0]
    
    # Calculate saturation (average of a and b channel stds)
    saturation = (stds[1] + stds[2]) / 2
    
    return {
        'contrast': contrast,
        'color_balance': color_balance,
        'brightness': brightness,
        'saturation': saturation,
        'overall_score': (contrast + saturation) / 2  # Simple overall score
    }

def smart_selection(image_files):
    """
    Analyze all images and return the best one to use as reference.
    
    This function implements a smart selection algorithm to choose the best reference image
    for stain normalization based on quality metrics. It analyzes each image and selects
    the one with the highest overall quality score.
    
    Args:
        image_files (list): List of Path objects pointing to image files
        
    Returns:
        tuple: (best_image, best_path) where best_image is the numpy array of the best image
               and best_path is the Path object pointing to the best image
        
    Raises:
        ValueError: If no suitable reference image is found
    """
    best_score = -float('inf')
    best_image = None
    best_path = None
    quality_metrics = {}
    
    print("\nAnalyzing images for reference selection...")
    for img_path in image_files:
        try:
            # Load and analyze image
            img = load_image(img_path)
            metrics = analyze_image_quality(img)
            quality_metrics[img_path.name] = metrics
            
            # Print metrics for this image
            print(f"\n{img_path.name}:")
            print(f"  Contrast: {metrics['contrast']:.2f}")
            print(f"  Color Balance: {metrics['color_balance']:.2f}")
            print(f"  Brightness: {metrics['brightness']:.2f}")
            print(f"  Saturation: {metrics['saturation']:.2f}")
            print(f"  Overall Score: {metrics['overall_score']:.2f}")
            
            # Update best image if this one is better
            if metrics['overall_score'] > best_score:
                best_score = metrics['overall_score']
                best_image = img
                best_path = img_path
                
        except Exception as e:
            print(f"Error analyzing {img_path.name}: {str(e)}")
            continue
    
    if best_image is None:
        raise ValueError("No suitable reference image found")
    
    print(f"\nSelected {best_path.name} as reference image")
    print("This image was chosen based on optimal contrast and color balance")
    
    return best_image, best_path

def visualize_normalization(original, normalized, title="Stain Normalization Comparison"):
    """
    Visualize original and normalized images side by side.
    
    This function creates a matplotlib figure showing the original and normalized
    images side by side for easy comparison.
    
    Args:
        original (numpy.ndarray): Original image in RGB format
        normalized (numpy.ndarray): Normalized image in RGB format
        title (str, optional): Title for the figure. Defaults to "Stain Normalization Comparison".
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(normalized)
    axes[1].set_title("Reinhard Normalized")
    axes[1].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def main():
    """
    Main function to run the stain normalization process.
    
    This function:
    1. Sets up the input and output directories
    2. Finds all images in the input directory
    3. Uses smart selection to choose the best reference image
    4. Normalizes all other images using the Reinhard method
    5. Saves the results to the output directory
    """
    # Setup paths
    base_dir = Path.cwd()  # Get current working directory
    test_images_dir = base_dir / "test_batch"
    output_dir = test_images_dir / "smart_selection_results"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Looking for images in: {test_images_dir}")
    print(f"Saving results to: {output_dir}")
    
    # Find all images
    image_files = list(test_images_dir.glob("*.jpg"))
    if not image_files:
        raise ValueError(f"No JPG images found in {test_images_dir}")
    
    print(f"Found {len(image_files)} images in test_batch directory")
    
    # Find the best reference image using smart selection
    reference_image, reference_path = smart_selection(image_files)
    
    # Initialize normalizer with the best reference image
    reinhard_normalizer = ReinhardNormalizer()
    reinhard_normalizer.fit(reference_image)
    
    # Process each image
    for img_path in image_files:
        if img_path == reference_path:  # Skip reference image
            continue
            
        print(f"\nProcessing: {img_path.name}")
        try:
            # Load image
            original = load_image(img_path)
            
            # Apply Reinhard normalization
            print("Applying Reinhard normalization...")
            normalized = reinhard_normalizer.normalize(original)
            
            # Visualize results
            print("Generating visualization...")
            fig = visualize_normalization(original, normalized,
                                    title=f"Stain Normalization - {img_path.name}")
            
            # Save visualization
            output_path = output_dir / f"normalized_{img_path.stem}.png"
            plt.savefig(output_path)
            plt.close()
            
            # Save normalized image
            Image.fromarray(normalized).save(
                output_dir / f"reinhard_{img_path.stem}.png")
            
            print(f"Successfully processed {img_path.name}")
            print(f"Results saved in {output_dir}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
