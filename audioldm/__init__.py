import os
import urllib.request

from .ldm import LatentDiffusion
from .pipeline import *
from .utils import get_time, save_wave, seed_everything

meta = {
    "audioldm": {
        "path": os.path.join(
            os.path.expanduser("~"),
            ".cache/audioldm/audioldm-s-full.ckpt",
        ),
        "url": "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1",
    },
}

if not os.path.exists(meta["audioldm"]["path"]):
    os.makedirs(os.path.dirname(meta["audioldm"]["path"]), exist_ok=True)
    print("Downloading the main structure of audioldm")

    urllib.request.urlretrieve(meta["audioldm"]["url"], meta["audioldm"]["path"])
    print(
        "Weights downloaded in: {} Size: {}".format(
            meta["audioldm"]["path"],
            os.path.getsize(meta["audioldm"]["path"]),
        )
    )
