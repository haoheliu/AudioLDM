import os

import argparse
import yaml
import torch
from torch import autocast
from tqdm import tqdm, trange

from audioldm import LatentDiffusion, seed_everything
from audioldm.utils import default_audioldm_config
from audioldm.audio import wav_to_fbank, TacotronSTFT
from audioldm.latent_diffusion.ddim import DDIMSampler
from einops import repeat

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


def build_model(
    ckpt_path=os.path.join(
        os.path.expanduser("~"),
        ".cache/audioldm/audioldm-s-full.ckpt",
    ),
    config=None,
):
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
    ddim_steps=200,
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
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform


def style_transfer(
    latent_diffusion,
    text,
    original_audio_file_path,
    transfer_strength,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    ddim_steps=200,
    config=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()

    seed_everything(int(seed))
    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    init_latent = latent_diffusion.get_first_stage_encoding(
        latent_diffusion.encode_first_stage(mel)
    )  # move to latent space, encode and sample

    sampler = DDIMSampler(latent_diffusion)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)

    t_enc = int(transfer_strength * ddim_steps)
    prompts = text

    with torch.no_grad():
        with autocast("cuda"):
            with latent_diffusion.ema_scope():
                uc = None
                if guidance_scale != 1.0:
                    uc = latent_diffusion.cond_stage_model.get_unconditional_condition(
                        batchsize
                    )

                c = latent_diffusion.get_learned_conditioning([prompts] * batchsize)

                z_enc = sampler.stochastic_encode(
                    init_latent, torch.tensor([t_enc] * batchsize).to(device)
                )

                samples = sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                )

                x_samples = latent_diffusion.decode_first_stage(samples)

                waveform = latent_diffusion.first_stage_model.decode_to_waveform(
                    x_samples
                )

    return waveform
