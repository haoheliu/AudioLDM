# Text-to-Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2109.13731-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaviinha/NeuralTextToAudio/blob/main/AudioLDM_pub.ipynb) [![Replicate](https://replicate.com/jagilley/audio-ldm/badge)](https://replicate.com/jagilley/audio-ldm)

<!-- # [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) -->

Generate speech, sound effects, music and beyond.

<hr>

## Important tricks to make your generated audio sound better

1. Try to use more adjectives to describe your sound. For example: "A man is speaking clearly and slowly in a professional studio" is better than "A man is speaking". This can make sure AudioLDM understand what you want.
2. Try to use different random seeds, which can affect the generation quality significantly sometimes.
3. It's best to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with, such as 'mummy'.

## Web APP

1. Prepare running environment

```shell
conda create -n audioldm python=3.8; conda activate audioldm
pip3 install audioldm==0.0.8
git clone https://github.com/haoheliu/AudioLDM; cd AudioLDM
```

2. Start the web application (powered by Gradio)

```shell
python3 app.py
```

3. A link will be printed out. Click the link to open the browser and play.

## Command Line Interface (CLI) Usage

1. Prepare running environment

```shell
# Optional
conda create -n audioldm python=3.8; conda activate audioldm
# Install AudioLDM
pip3 install audioldm==0.0.8
```

2. text-to-audio generation

```python
# Test run
audioldm -t "A hammer is hitting a wooden surface" # The default --mode is "generation"
```

3. audio-to-audio style transfer

```python
# Test run
# --file_path is the original audio file for transfer
# -t is the text AudioLDM uses for transfer.
# Please make sure that --file_path exist
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing"

# Tune the value of --transfer_strength is important!
# --transfer_strength: A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text
audioldm --mode "transfer" --file_path trumpet.wav -t "Children Singing" --transfer_strength 0.25
```

For more options on guidance scale, batchsize, seed, ddim steps, etc., please run

```shell
audioldm -h
```

For the evaluation of audio generative model, please refer to [audioldm_eval](https://github.com/haoheliu/audioldm_eval).

## Web Demo

Integrated into [Hugging Face Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)

# TODO

- [ ] Update the checkpoint with more training steps.
- [ ] Add AudioCaps finetuned AudioLDM-S model
- [x] Build pip installable package for commandline use
- [x] Build Gradio web application
- [x] Add text-guided style transfer
- [ ] Add audio super-resolution
- [ ] Add audio inpainting

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

## Hardware requirement

- GPU with 8GB of dedicated VRAM
- A system with a 64-bit operating system (Windows 7, 8.1 or 10, Ubuntu 16.04 or later, or macOS 10.13 or later) 16GB or more of system RAM

## Reference

Part of the code is borrowed from the following repos. We would like to thank the authors of these repos for their contribution.

> <https://github.com/LAION-AI/CLAP>

> <https://github.com/CompVis/stable-diffusion>

> <https://github.com/v-iashin/SpecVQGAN>

> <https://github.com/toshas/torch-fidelity>

We build the model with data from AudioSet, Freesound and BBC Sound Effect library. We share this demo based on the UK copyright exception of data for academic research.

<!-- This code repo is strictly for research demo purpose only. For commercial use please contact us. -->
