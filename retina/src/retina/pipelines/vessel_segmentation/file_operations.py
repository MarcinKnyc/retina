import os
import tarfile
import zipfile
import gzip
import shutil
import glob
from PIL import Image


def rename_folder(old_path: str, new_path: str) -> None:
    """
    Renames a folder from old_path to new_path.

    Parameters:
    old_path (str): The current path of the folder.
    new_path (str): The new path of the folder.

    Returns:
    None
    """
    try:
        os.rename(old_path, new_path)
        print(f"Folder renamed from '{old_path}' to '{
              new_path}' successfully.")
    except FileNotFoundError:
        print(f"The folder at '{old_path}' does not exist.")
    except FileExistsError:
        print(f"A folder already exists at the new path '{new_path}'.")
    except PermissionError:
        print(f"Permission denied to rename the folder.")
    except Exception as e:
        print(f"An error occurred: {e}")


def unzip(url: str, output: str):
    with zipfile.ZipFile(url, 'r') as zip_ref:
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
