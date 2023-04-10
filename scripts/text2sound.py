import os
from audioldm import text_to_audio, build_model, save_wave

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-t",
    "--text",
    type=str,
    required=False,
    default="A hammer is hitting a wooden surface",
    help="Text prompt to the model for audio generation",
)

parser.add_argument(
    "-s",
    "--save_path",
    type=str,
    required=False,
    help="The path to save model output",
    default="./output",
)

parser.add_argument(
    "-ckpt",
    "--ckpt_path",
    type=str,
    required=False,
    help="The path to the pretrained .ckpt model",
    default="./ckpt/audioldm-s-full.ckpt",
)

parser.add_argument(
    "-b",
    "--batchsize",
    type=int,
    required=False,
    default=1,
    help="Generate how many samples at the same time",
)

parser.add_argument(
    "-gs",
    "--guidance_scale",
    type=float,
    required=False,
    default=2.5,
    help="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)",
)

parser.add_argument(
    "-dur",
    "--duration",
    type=float,
    required=False,
    default=10.0,
    help="The duration of the samples",
)

parser.add_argument(
    "-n",
    "--n_candidate_gen_per_text",
    type=int,
    required=False,
    default=3,
    help="Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation",
)

parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
    help="Change this value (any integer number) will lead to a different generation result.",
)

args = parser.parse_args()

assert args.duration % 2.5 == 0, "Duration must be a multiple of 2.5"

save_path = args.save_path
text = args.text
random_seed = args.seed
duration = args.duration
guidance_scale = args.guidance_scale
n_candidate_gen_per_text = args.n_candidate_gen_per_text

os.makedirs(save_path, exist_ok=True)
audioldm = build_model(ckpt_path=args.ckpt_path)
waveform = text_to_audio(
    audioldm,
    text,
    seed=random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    n_candidate_gen_per_text=n_candidate_gen_per_text,
    batchsize=args.batchsize,
)

save_wave(waveform, save_path, name=text)
