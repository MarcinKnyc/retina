import os
import zipfile


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
        print(f"Folder renamed from '{old_path}' to '{new_path}' successfully.")
    except FileNotFoundError:
        print(f"The folder at '{old_path}' does not exist.")
    except FileExistsError:
        print(f"A folder already exists at the new path '{new_path}'.")
    except PermissionError:
        print(f"Permission denied to rename the folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

def unzip_folder(zipped_images_path: str, extracted_images_path: str):    
    with zipfile.ZipFile(zipped_images_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_images_path)