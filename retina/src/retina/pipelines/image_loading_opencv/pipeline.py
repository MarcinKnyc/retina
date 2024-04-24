import cv2
import os
import zipfile
from typing import List
from kedro.pipeline import node, Pipeline

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

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=unzip_folder,
                inputs=["params:zipped_images_path", "params:extracted_images_path"],
                outputs=None,
                name="unzip_folder"
            ),
            node(
                func=load_images_from_directory,
                inputs=["params:extracted_images_path", "params:valid_extensions"],
                outputs="loaded_images",
                name="load_images"
            )
        ]
    )
