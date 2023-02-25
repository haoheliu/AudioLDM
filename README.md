# :sound: Audio Generation with AudioLDM

[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaviinha/NeuralTextToAudio/blob/main/AudioLDM_pub.ipynb?force_theme=dark)  [![Replicate](https://replicate.com/jagilley/audio-ldm/badge)](https://replicate.com/jagilley/audio-ldm)

<!-- # [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) -->

**Generate speech, sound effects, music and beyond.**

This repo currently suppport: 

- **Text-to-Audio Generation**: Generate audio given text input.
- **Audio-to-Audio Generation**: Given an audio, generate another audio that contain the same type of sound. 
- **Text-guided Audio-to-Audio Style Transfer**: Transfer the sound of an audio into another one using the text description.

<hr>

## Important tricks to make your generated audio sound better
1. Try to provide more hints to AudioLDM, such as using more adjectives to describe your sound (e.g., clearly, high quality) or make your target more specific (e.g., "water stream in a forest" instead of "stream"). This can make sure AudioLDM understand what you want. 
2. Try to use different random seeds, which can affect the generation quality significantly sometimes.
3. It's best to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with.

# Change Log

**2023-02-24**: Add audio-to-audio generation. Add test cases. Add a pipeline (python function) for audio super-resolution and inpainting.

**2023-02-15**: Add audio style transfer. Add more options on generation.

## Web APP

The web APP currently only support Text-to-Audio generation. For full functionality please refer to the [Commandline Usage](https://github.com/haoheliu/AudioLDM#commandline-usage).

1. Prepare running environment
```shell
conda create -n audioldm python=3.8; conda activate audioldm
pip3 install audioldm
git clone https://github.com/haoheliu/AudioLDM; cd AudioLDM
```
2. Start the web application (powered by Gradio)
```shell
python3 app.py
```
3. A link will be printed out. Click the link to open the browser and play.

## Commandline Usage
Prepare running environment
```shell
# Optional
conda create -n audioldm python=3.8; conda activate audioldm
# Install AudioLDM
pip3 install audioldm
```

:star2: **Text-to-Audio Generation**: generate an audio guided by a text
```shell
# The default --mode is "generation"
audioldm -t "A hammer is hitting a wooden surface" 
# Result will be saved in "./output/generation"
```

:star2: **Audio-to-Audio Generation**: generate an audio guided by an audio (output will have similar audio events as the input audio file).
```shell
audioldm --file_path trumpet.wav
# Result will be saved in "./output/generation_audio_to_audio/trumpet"
```

:star2: **Text-guided Audio-to-Audio Style Transfer**
```shell
# Test run
# --file_path is the original audio file for transfer
# -t is the text AudioLDM uses for transfer. 
# Please make sure that --file_path exist
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" 
# Result will be saved in "./output/transfer/trumpet"

# Tune the value of --transfer_strength is important!
# --transfer_strength: A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" --transfer_strength 0.25
```

For more options on guidance scale, batchsize, seed, ddim steps, etc., please run
```shell
audioldm -h
```
```console
usage: audioldm [-h] [--mode {generation,transfer}] [-t TEXT] [-f FILE_PATH] [--transfer_strength TRANSFER_STRENGTH] [-s SAVE_PATH] [-ckpt CKPT_PATH] [-b BATCHSIZE] [--ddim_steps DDIM_STEPS] [-gs GUIDANCE_SCALE]
                [-dur DURATION] [-n N_CANDIDATE_GEN_PER_TEXT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --mode {generation,transfer}
                        generation: text-to-audio generation; transfer: style transfer. DEFAULT "generation"
  -t TEXT, --text TEXT  Text prompt to the model for audio generation. DEFAULT ""
  -f FILE_PATH, --file_path FILE_PATH
                        (--mode transfer): Original audio file for style transfer; Or (--mode generation): the guidance audio file for generating simialr audio. DEFAULT None
  --transfer_strength TRANSFER_STRENGTH
                        A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text. DEFAULT 0.5
  -s SAVE_PATH, --save_path SAVE_PATH
                        The path to save model output. DEFAULT "./output"
  -ckpt CKPT_PATH, --ckpt_path CKPT_PATH
                        The path to the pretrained .ckpt model. DEFAULT "~/.cache/audioldm/audioldm-s-full.ckpt"
  -b BATCHSIZE, --batchsize BATCHSIZE
                        Generate how many samples at the same time. DEFAULT 1
  --ddim_steps DDIM_STEPS
                        The sampling step for DDIM. DEFAULT 200
  -gs GUIDANCE_SCALE, --guidance_scale GUIDANCE_SCALE
                        Guidance scale (Large => better relavancy to text; Small => better diversity). DEFAULT 2.5
  -dur DURATION, --duration DURATION
                        The duration of the samples. DEFAULT 10
  -n N_CANDIDATE_GEN_PER_TEXT, --n_candidate_gen_per_text N_CANDIDATE_GEN_PER_TEXT
                        Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier
                        computation. DEFAULT 3
  --seed SEED           Change this value (any integer number) will lead to a different generation result. DEFAULT 42
```



For the evaluation of audio generative model, please refer to [audioldm_eval](https://github.com/haoheliu/audioldm_eval).

# Web Demo

Integrated into [Hugging Face Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)

# TODO

- [ ] Update the checkpoint with more training steps.
- [ ] Add AudioCaps finetuned AudioLDM-S model
- [x] Build pip installable package for commandline use
- [x] Build Gradio web application
- [x] Add text-guided style transfer
- [x] Add audio-to-audio generation
- [x] Add audio super-resolution
- [x] Add audio inpainting

## Cite this work

If you found this tool useful, please consider citing
```bibtex
@article{liu2023audioldm,
  title={AudioLDM: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={arXiv preprint arXiv:2301.12503},
  year={2023}
}
```

# Hardware requirement
- GPU with 8GB of dedicated VRAM
- A system with a 64-bit operating system (Windows 7, 8.1 or 10, Ubuntu 16.04 or later, or macOS 10.13 or later) 16GB or more of system RAM

## Reference
Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution. 

> https://github.com/LAION-AI/CLAP

> https://github.com/CompVis/stable-diffusion

> https://github.com/v-iashin/SpecVQGAN 

> https://github.com/toshas/torch-fidelity


We build the model with data from AudioSet, Freesound and BBC Sound Effect library. We share this demo based on the UK copyright exception of data for academic research. 

<!-- This code repo is strictly for research demo purpose only. For commercial use please contact us. -->
