import os
from urllib.request import urlretrieve as download
from .file_operations import untar, ungz, rename, split_data, ppm2png, clean_ppm


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


def prepare_stare(stare_images_url: str, stare_labels_url: str, datapath: str, name: str) -> int:
    def make_paths(path: str):
        os.makedirs(path)
        os.makedirs(path + "train/image")
        os.makedirs(path + "train/mask")
        os.makedirs(path + "test/image")
        os.makedirs(path + "test/mask")

    path = datapath + name
    if os.path.exists(path):
        return 0

    make_paths(path)

    get(stare_images_url, path, "image")
    get(stare_labels_url, path, "mask")

    split_data(path + 'image', path + 'train/image', path + 'test/image')
    split_data(path + 'mask', path + 'train/mask', path + 'test/mask')

    ppm2png(path)
    clean_ppm(path)

    rename(path + "train/image")
    rename(path + "train/mask")
    rename(path + "test/image")
    rename(path + "test/mask")

    return 0
