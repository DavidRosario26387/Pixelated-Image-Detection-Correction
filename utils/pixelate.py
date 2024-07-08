import os
import random
import cv2
from PIL import Image
import numpy as np

def pixelate_image(image, scale_factor, interpolation):
    """
    Pixelate the given image by downscaling and then upscaling.
    Args:
    - image: The input image.
    - scale_factor: The factor by which to downscale and then upscale the image.
    - interpolation: The interpolation method to use.
    Returns:
    - The pixelated image.
    """
    height, width = image.shape[:2]
    # Downscale the image
    downscaled_image = cv2.resize(image, (width // scale_factor, height // scale_factor), interpolation=interpolation)
    # Upscale the image back to original size
    pixelated_image = cv2.resize(downscaled_image, (width, height), interpolation=interpolation)
    return pixelated_image

def process_images(input_folder, output_folder, scale_factors=[5, 6]):
    """
    Process images in the input_folder, pixelate them, and save to output_folder.
    Args:
    - input_folder: Path to the folder containing input images.
    - output_folder: Path to the folder to save pixelated images.
    - scale_factors: List of scale factors to randomly choose from for pixelation.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Unable to read {image_path}")
                continue

            # Randomly choose scale factor and interpolation method
            scale_factor = random.choice(scale_factors)
            interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])

            # Pixelate the image
            pixelated_image = pixelate_image(image, scale_factor, interpolation)

            # Save the pixelated image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, pixelated_image)

# Example usage:
input_folder = 'path/to/input/folder'
output_folder = 'path/to/output/folder'
process_images(input_folder, output_folder)
