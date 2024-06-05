from tqdm import tqdm
import cv2
import numpy as np

from skimage.measure import moments_central, moments_hu, moments_normalized
from skimage.util import view_as_windows


def create_feature_vector(image):
    green_channel = image[:, :, 1]
    inverted_green = cv2.bitwise_not(green_channel)

    grad_orientation = compute_gradient_orientation(green_channel)
    morph_transformation = morphological_top_hat(green_channel)
    line_strength_1 = line_strength(green_channel)
    line_strength_2 = line_strength(cv2.bitwise_not(green_channel))
    gabor_response = gabor_filter_responses(inverted_green)

    features = np.stack([
        grad_orientation,
        morph_transformation,
        line_strength_1,
        line_strength_2,
        gabor_response,
        inverted_green,
    ], axis=-1)

    return features


def compute_gradient_orientation(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    ux = np.divide(gx, magnitude, out=np.zeros_like(gx), where=magnitude != 0)
    uy = np.divide(gy, magnitude, out=np.zeros_like(gy), where=magnitude != 0)
    return np.arctan2(uy, ux)


def morphological_top_hat(image):
    se_length = 21
    top_hat = np.zeros_like(image, dtype=np.float64)
    for angle in range(0, 180, 22):
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_length, 1))
        rotated_se = cv2.warpAffine(se, cv2.getRotationMatrix2D(
            (se_length//2, 0), angle, 1), (se_length, se_length))
        top_hat += cv2.morphologyEx(image, cv2.MORPH_TOPHAT, rotated_se)
    return top_hat


def line_strength(image):
    lines = np.zeros_like(image, dtype=np.float64)
    se_length = 15
    for angle in range(0, 180, 15):
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_length, 1))
        rotated_se = cv2.warpAffine(se, cv2.getRotationMatrix2D(
            (se_length//2, 0), angle, 1), (se_length, se_length))
        lines = np.maximum(lines, cv2.filter2D(image, cv2.CV_64F, rotated_se))
    return lines


def gabor_filter_responses(image):
    gabor_features = []
    for theta in range(0, 180, 10):
        theta_rad = theta * np.pi / 180
        for sigma in (2, 3, 4, 5):
            gabor_kernel = cv2.getGaborKernel(
                (sigma*6, sigma*6), sigma, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_64F)
            gabor_features.append(cv2.filter2D(
                image, cv2.CV_64F, gabor_kernel))
    return np.stack(gabor_features, axis=-1).max(axis=-1)


def create_dataset(photos: list, masks: list) -> tuple:
    """Creates a dataset by extracting patches and features from images."""
    ds_photos = []
    ds_masks = []

    for photo, mask in tqdm(list(zip(photos, masks)), total=len(photos), desc='Processing photos and their masks'):
        feature_vector = create_feature_vector(photo)
        height, width, num_features = feature_vector.shape
        # Flatten the features
        flattened_features = feature_vector.reshape(-1, num_features)
        ds_photos.append(flattened_features)
        ds_masks.append(mask.flatten())  # Flatten the mask

    ds_photos = np.vstack(ds_photos)
    ds_masks = np.hstack(ds_masks)

    return ds_photos, ds_masks
