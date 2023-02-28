from .ldm import LatentDiffusion
from .utils import seed_everything, save_wave, get_time, get_duration
from .pipeline import *

import os
import urllib.request
import progressbar


CACHE_DIR = os.getenv(
    "AUDIOLDM_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache/audioldm"))


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()
            
meta = {
    "audioldm": {
        "path": os.path.join(
            CACHE_DIR,
            "audioldm-s-full.ckpt",
        ),
        "url": "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1",
    },
}

if not os.path.exists(meta["audioldm"]["path"]) or os.path.getsize(meta["audioldm"]["path"]) < 2*10**9:
    os.makedirs(os.path.dirname(meta["audioldm"]["path"]), exist_ok=True)
    print(f"Downloading the main structure of audioldm into {os.path.dirname(meta['audioldm']['path'])}")

    urllib.request.urlretrieve(meta["audioldm"]["url"], meta["audioldm"]["path"], MyProgressBar())
    print(
        "Weights downloaded in: {} Size: {}".format(
            meta["audioldm"]["path"],
            os.path.getsize(meta["audioldm"]["path"]),
        )
    )
