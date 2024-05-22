import cv2
import os
import zipfile
from typing import List, Tuple
from kedro.pipeline import node, Pipeline
import numpy as np

def unzip_folder(zipped_images_path: str, extracted_images_path: str):    
    with zipfile.ZipFile(zipped_images_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_images_path)

def load_images_from_directory(extracted_images_path: str, valid_extensions: List[str]):
    images = []
    warnings = []
    for root, dirs, files in os.walk(extracted_images_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                else:
                    warnings.append(f"Failed to load: {file_path}")
            else:
                warnings.append(f"Unsupported file extension: {file_path}")
    
    for warning in warnings:
        print("Warning:", warning)
    
    return images

def extract_example_pixel_by_pixel_feature_vectors(images: List) -> List[np.ndarray]:
    feature_vectors = []
    
    for image in images:
        height, width, channels = image.shape
        
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create feature vector for each pixel
        features = np.zeros((height, width, 10))
        
        for y in range(height):
            for x in range(width):
                # Extract RGB values
                b, g, r = image[y, x]
                
                # Compute grayscale intensity
                intensity = gray[y, x]
                
                # Compute average color value
                avg_color = np.mean([r, g, b])
                
                # Compute standard deviation of color values
                std_color = np.std([r, g, b])
                
                # Compute min and max color values
                min_color = np.min([r, g, b])
                max_color = np.max([r, g, b])
                
                # Gradient magnitudes
                grad_mag_x = grad_x[y, x]
                grad_mag_y = grad_y[y, x]
                
                # Combine features into a vector
                features[y, x] = [r, g, b, intensity, avg_color, std_color, min_color, max_color, grad_mag_x, grad_mag_y]
        
        feature_vectors.append(features)
    
    return feature_vectors

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=unzip_folder,
                inputs=["params:kaggle_zipped_images_path", "params:kaggle_extracted_images_path"],
                outputs=None,
                name="unzip_folder"
            ),
            node(
                func=load_images_from_directory,
                inputs=["params:kaggle_extracted_images_path", "params:valid_extensions"],
                outputs="loaded_images",
                name="load_images"
            ),
            # node(
            #     func=extract_example_pixel_by_pixel_feature_vectors,
            #     inputs=["loaded_images"],
            #     outputs="pix_by_pix_feature_vectors",
            #     name="extract_example_pixel_by_pixel_feature_vectors"
            # )
        ]
    )
