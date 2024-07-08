from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_crop_image(img_path):
    # Load image using keras.preprocessing.image.load_img
    img = image.load_img(img_path)
    
    # Get dimensions of the loaded image
    width, height = img.size
    
    # Calculate the coordinates for cropping a central 224x224 area
    left = (width - 224) // 2
    top = (height - 224) // 2
    right = left + 224
    bottom = top + 224
    
    # Crop the image to the specified size
    img = img.crop((left, top, right, bottom))
    
    # Convert the image to RGB mode (if not already)
    img = img.convert('RGB')
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # Add batch dimension (since model expects a batch of images)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

# Example usage:
if __name__ == "__main__":
    img_path = 'path_to_your_image.jpg'  # Replace with your image path
    cropped_img = load_and_crop_image(img_path)
    # The function now returns cropped_img without printing anything
