import os
import tarfile
import zipfile
import gzip
import shutil
import glob
import cv2

import numpy as np
from PIL import Image


def load_data(input_path: str, debug: bool) -> tuple:
    """Loads and preprocesses the train and test datasets."""
    train_photo_paths = [os.path.join(input_path, 'train/image', x)
                         for x in os.listdir(os.path.join(input_path, 'train/image'))]

    train_label_paths = [os.path.join(input_path, 'train/mask', x)
                         for x in os.listdir(os.path.join(input_path, 'train/mask'))]

    train_photos = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                    for path in train_photo_paths]

    train_masks = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
                   for path in train_label_paths]
    train_masks = [np.where(mask > 0, 255, 0)
                   for mask in train_masks]

    test_photo_paths = [os.path.join(input_path, 'test/image', x)
                        for x in os.listdir(os.path.join(input_path, 'test/image'))]

    test_label_paths = [os.path.join(input_path, 'test/mask', x)
                        for x in os.listdir(os.path.join(input_path, 'test/mask'))]

    test_photos = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
                   for path in test_photo_paths]

    test_masks = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
                  for path in test_label_paths]
    test_masks = np.array(test_masks).astype(np.uint8)
    test_masks = [np.where(mask > 0, 255, 0)
                  for mask in test_masks]

    if debug:
        return train_photos[:5], train_masks[:5], test_photos[:5], test_masks[:5]

    return train_photos, train_masks, test_photos, test_masks


def unzip(file: str, output: str):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(output)


def untar(input: str, output: str) -> None:
    with tarfile.open(input, "r") as tar:
        tar.extractall(output)


def ungz(path: str) -> None:
    for filename in os.listdir(path):
        if filename.endswith('.gz'):
            filepath = os.path.join(path, filename)
            output_filepath = os.path.join(
                path, filename[:-3])  # Remove the .gz extension

            with gzip.open(filepath, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def split_data(input: str, output: str, output2: str, amount: int) -> None:
    for i, file in enumerate(os.listdir(input)):
        if i >= amount:
            break
        shutil.copy(input + "/" + file, output + "/" + file)
    for file in os.listdir(input):
        shutil.copy(input + "/" + file, output2 + "/" + file)


def format2png(path: str, format: str) -> None:
    for filepath in glob.iglob(path + "/**/*." + format + "*", recursive=True):
        with Image.open(filepath) as im:
            filepath2 = filepath[:-4]
            filepath2 = ''.join(ch for ch in filepath2)
            filepath2 = filepath2 + ".png"

            im.save(filepath2)


def clean_format(path: str, format: str) -> None:
    for filepath in glob.iglob(path + "/**/*." + format + "*", recursive=True):
        os.remove(filepath)


def rename(path: str) -> None:
    for i, filename in enumerate(os.listdir(path)):
        os.rename(path + "/" + filename, path + "/" + str(i) + filename[-4:])


def rename_all(path: str) -> None:
    rename(path + "train/image")
    rename(path + "train/mask")
    rename(path + "test/image")
    rename(path + "test/mask")
