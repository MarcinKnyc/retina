from tqdm import tqdm
import cv2
import numpy as np

from skimage.measure import moments_central, moments_hu, moments_normalized
from skimage.util import view_as_windows

def preprocess_images(train_raw_photos: list, test_raw_photos: list) -> tuple:
    """Preprocesses the train and test images by enhancing contrast."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    train_photos_green = [photo[:, :, 1] for photo in train_raw_photos]    
    # train_photos_clahe = [clahe.apply(photo) for photo in train_photos_green]

    test_photos_green = [photo[:, :, 1] for photo in test_raw_photos]
    # test_photos = [clahe.apply(photo) for photo in test_photos_green]

    return train_photos_green, test_photos_green

def features_extracting(photo_batch: np.ndarray) -> np.ndarray:
    """Extracts features from image patches."""
    color_variance = np.var(photo_batch, axis=(1, 2))
    hu_moments = np.zeros((len(photo_batch), 7))
    for i, photo in enumerate(photo_batch):
        mu = moments_central(photo, order=5)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        hu_moments[i] = hu

    features = np.concatenate((color_variance.reshape(-1, 1), hu_moments), axis=1)
    return features




def create_dataset(photos: list, masks: list, size: int = 5, step: int = 1) -> tuple:
    """Creates a dataset by extracting patches and features from images."""
    ds_photos = []
    ds_masks = []
    for photo, mask in tqdm(zip(photos, masks), total=len(photos), desc='Processing photos and their masks'):
        photo_pad = np.pad(photo, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant')
        mask_pad = np.pad(mask, ((size // 2, size // 2), (size // 2, size // 2)), mode='constant')

        patches_photo = view_as_windows(photo_pad, (size, size), step=step).reshape(-1, size, size)
        patches_mask = view_as_windows(mask_pad, (size, size), step=step).reshape(-1, size, size)

        batched_features = features_extracting(patches_photo)

        ds_photos.extend(batched_features)
        ds_masks.extend(patches_mask[:, size // 2, size // 2])

    return ds_photos, ds_masks