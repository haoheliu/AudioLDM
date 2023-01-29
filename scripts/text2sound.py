import sys

# sys.path.append(
#     "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/AudioLDM-python"
# )

import os

import argparse
import yaml
import torch

from audioldm.model.pipline import LatentDiffusion
from audioldm.utils import seed_everything

import time

t = time.localtime()
current_time = time.strftime("%d_%m_%Y_%H_%M_%S", t)

def make_batch_for_text_to_audio(text, batchsize=2):
    text = [text] * batchsize
    if batchsize < 2:
        print("Warning: Batchsize must be at least 2. Batchsize is set to 2.")
    fbank = torch.zeros((batchsize, 1024, 64))  # Not used
    stft = torch.zeros((batchsize, 1024, 512))  # Not used
    waveform = torch.zeros((batchsize, 160000))  # Not used
    fname = ["%s.wav" % x for x in range(batchsize)]
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )  # fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text
    return batch

def main(args, config):
    log_path = config["log_directory"]

    config["id"]["version"] = "%s_%s" % (config["id"]["name"], config["id"]["version"])

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    latent_diffusion.set_log_dir(log_path, "audioverse", config["id"]["version"])

    checkpoint_path = os.path.join(
        log_path, "audioverse", config["id"]["version"], "checkpoints"
    )
    os.makedirs(checkpoint_path, exist_ok=True)

    resume_from_checkpoint = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/AudioLDM-python/ckpt/ldm_trimmed.ckpt"

    checkpoint = torch.load(resume_from_checkpoint)
    latent_diffusion.load_state_dict(checkpoint["state_dict"])

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    latent_diffusion.cond_stage_model.embed_mode = "text"

    batch = make_batch_for_text_to_audio(args.text)

    with torch.no_grad():
        latent_diffusion.generate_sample(
            [batch],
            name="text2sound/%s_%s" % (current_time, str(args.text)[:77]),
            unconditional_guidance_scale=2.5,
            n_gen=1,
        )

if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=False,
        help="path to config file",
        default="config.yaml",
    )

    parser.add_argument(
        "-t", "--text", type=str, required=False, default="Ambient music"
    )

    args = parser.parse_args()

    config = args.config

    config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)

    main(args, config)
