import os

import argparse
import yaml
import torch

from audioldm import LatentDiffusion, seed_everything
from audioldm.utils import default_audioldm_config

import time


def make_batch_for_text_to_audio(text, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    stft = torch.zeros((batchsize, 1024, 512))  # Not used
    waveform = torch.zeros((batchsize, 160000))  # Not used
    fname = [""] * batchsize  # Not used
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )
    return batch


def build_model(ckpt_path=os.path.join(
                os.path.expanduser("~"),
                ".cache/audioldm/audioldm-s-full.ckpt",
                ), config=None):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"
    return latent_diffusion


def duration_to_latent_t_size(duration):
    return int(duration * 25.6)


def text_to_audio(
    latent_diffusion,
    text,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    config=None,
):
    seed_everything(int(seed))
    batch = make_batch_for_text_to_audio(text, batchsize=batchsize)

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform
