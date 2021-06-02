import os
import urllib.request

import progressbar
import gdown


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


MODEL_PATH_ROOT_ = os.path.expanduser("~/.cache/ozeu/models")

def get_model_file(name, url):
    filepath = os.path.join(MODEL_PATH_ROOT_, name)
    if not os.path.exists(filepath):
        os.makedirs(MODEL_PATH_ROOT_, exist_ok=True)
        urllib.request.urlretrieve(url, filepath, show_progress)

    return filepath

def get_model_file_from_gdrive(name, url):
    filepath = os.path.join(MODEL_PATH_ROOT_, name)
    if not os.path.exists(filepath):
        os.makedirs(MODEL_PATH_ROOT_, exist_ok=True)
        gdown.download(url, filepath)

    return filepath