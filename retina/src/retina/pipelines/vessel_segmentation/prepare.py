import os
import glob
from urllib.request import urlretrieve as download
from shutil import rmtree
from .file_operations import untar, ungz, rename_all, split_data, format2png, clean_format, unzip


def make_paths(path: str):
    os.makedirs(path)
    os.makedirs(path + "train/image")
    os.makedirs(path + "train/mask")
    os.makedirs(path + "test/image")
    os.makedirs(path + "test/mask")


def split(path: str, src: str, amount: int) -> None:
    split_data(path + src,
               path + 'train/image',
               path + 'test/image', amount)
    split_data(path + 'mask',
               path + 'train/mask',
               path + 'test/mask', amount)
    rmtree(path + src)
    rmtree(path + 'mask')


def prepare_stare(stare_images_url: str, stare_labels_url: str, datapath: str, name: str) -> int:
    def get(url: str, path: str, name: str) -> None:
        tar = path + "tar.tar"

        try:
            download(url, tar)
        except IOError:
            raise IOError()

        extracted = path + "extracted"

        untar(tar, extracted)
        os.remove(tar)

        ungz(extracted)

        for item in os.listdir(extracted):
            if item.endswith(".gz"):
                os.remove(extracted + "/" + item)

        os.rename(extracted, path + name)

    path = datapath + name
    if os.path.exists(path):
        return 0

    make_paths(path)

    get(stare_images_url, path, "image")
    get(stare_labels_url, path, "mask")

    split(path, 'image', len(os.listdir(path + 'image')))

    format2png(path, "ppm")
    clean_format(path, "ppm")

    rename_all(path)

    return 0


def prepare_drive(datapath: str, name: str) -> int:
    path = datapath + name
    if os.path.exists(path):
        return 0

    unzip(path[:-1] + ".zip", datapath)

    os.makedirs(path + "train")

    rmtree(path + "training/mask")
    rmtree(path + "test/mask")
    rmtree(path + "test/2nd_manual")

    os.rename(path + "test/1st_manual", path + "test/mask")
    os.rename(path + "training/1st_manual", path + "train/mask")

    os.rename(path + "test/images", path + "test/image")
    os.rename(path + "training/images", path + "train/image")

    rmtree(path + "training")

    format2png(path, "tif")
    format2png(path, "gif")
    clean_format(path, "tif")
    clean_format(path, "gif")

    rename_all(path)

    return 0


def prepare_chasedb1(url: str, datapath: str, name: str) -> int:
    def get(url: str, path: str, name: str) -> None:
        zip = path + "zip.zip"

        try:
            download(url, zip)
        except IOError:
            raise IOError()

        extracted = path + "extracted"

        unzip(zip, extracted)
        os.remove(zip)

        os.rename(extracted, path + name)

    path = datapath + name
    if os.path.exists(path):
        return 0

    make_paths(path)

    get(url, path, "data")

    for filepath in glob.iglob(path + "/**/*_2ndHO.*", recursive=True):
        os.remove(filepath)

    format2png(path, "jpg")
    clean_format(path, "jpg")

    os.makedirs(path + "mask")
    for filepath in glob.iglob(path + "**/*_1stHO.*", recursive=True):
        os.rename(filepath, path + "mask/" +
                  filepath.replace("\\", "/").split("/")[-1])

    amount = 8
    split(path, 'data', amount)
    p = path + "test/image"
    for i, file in enumerate(os.listdir(p)):
        if i >= amount:
            break
        os.remove(p + "/" + file)
    p = path + "test/mask"
    for i, file in enumerate(os.listdir(p)):
        if i >= amount:
            break
        os.remove(p + "/" + file)

    rename_all(path)

    return 0
