import tarfile
import os
import gzip
import shutil
import glob
from PIL import Image


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


def split_data(input: str, output: str, output2: str) -> None:
    files = os.listdir(input)
    for file in files[:len(files)//2]:
        os.rename(input + "/" + file, output + "/" + file)
    for file in os.listdir(input):
        os.rename(input + "/" + file, output2 + "/" + file)
    shutil.rmtree(input)


def ppm2png(path: str) -> None:
    for filepath in glob.iglob(path + "/**/*.ppm*", recursive=True):
        with Image.open(filepath) as im:
            filepath2 = filepath[:-4]
            filepath2 = ''.join(ch for ch in filepath2)
            filepath2 = filepath2 + ".png"

            im.save(filepath2)


def clean_ppm(path: str) -> None:
    for filepath in glob.iglob(path + "/**/*.ppm*", recursive=True):
        os.remove(filepath)


def rename(path):
    for i, filename in enumerate(os.listdir(path)):
        os.rename(path + "/" + filename, path + "/" + str(i) + filename[-4:])
