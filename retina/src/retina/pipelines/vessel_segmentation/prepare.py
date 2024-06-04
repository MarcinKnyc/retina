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


def prepare_stare(stare_images_url: str, stare_labels_url: str, datapath: str) -> int:
    def make_paths(path: str):
        os.makedirs(path)
        os.makedirs(path + "train/image")
        os.makedirs(path + "train/mask")
        os.makedirs(path + "test/image")
        os.makedirs(path + "test/mask")

    if os.path.exists(datapath):
        return 0

    make_paths(datapath)

    get(stare_images_url, datapath, "image")
    get(stare_labels_url, datapath, "mask")

    split_data(datapath + 'image', datapath + 'train/image', datapath + 'test/image')
    split_data(datapath + 'mask', datapath + 'train/mask', datapath + 'test/mask')

    ppm2png(datapath)
    clean_ppm(datapath)

    rename(datapath + "train/image")
    rename(datapath + "train/mask")
    rename(datapath + "test/image")
    rename(datapath + "test/mask")

    return 0
