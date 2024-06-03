import cv2
import os
import numpy as np


def load_data(input_path: str, debug: bool) -> tuple:
    """Loads and preprocesses the train and test datasets."""
    train_photo_paths = [os.path.join(input_path, 'train/image', x) for x in os.listdir(os.path.join(input_path, 'train/image'))]
    train_label_paths = [os.path.join(input_path, 'train/mask', x) for x in os.listdir(os.path.join(input_path, 'train/mask'))]

    train_raw_photos = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in train_photo_paths]
    train_masks = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY) for path in train_label_paths]
    train_masks = [np.where(mask > 0, 255, 0) for mask in train_masks]

    test_photo_paths = [os.path.join(input_path, 'test/image', x) for x in os.listdir(os.path.join(input_path, 'test/image'))]
    test_label_paths = [os.path.join(input_path, 'test/mask', x) for x in os.listdir(os.path.join(input_path, 'test/mask'))]

    test_raw_photos = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in test_photo_paths]
    test_masks = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY) for path in test_label_paths]
    test_masks = np.array(test_masks).astype(np.uint8)
    test_masks = [np.where(mask > 0, 255, 0) for mask in test_masks]

    if debug:
        return train_raw_photos[:5], train_masks[:5], test_raw_photos[:5], test_masks[:5]
    
    return train_raw_photos, train_masks, test_raw_photos, test_masks


    