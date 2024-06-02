# :sound: Audio Generation with AudioLDM

[![arXiv](https://img.shields.io/badge/arXiv-2301.12503-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2301.12503)  [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://audioldm.github.io/)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/olaviinha/NeuralTextToAudio/blob/main/AudioLDM_pub.ipynb?force_theme=dark)  [![Replicate](https://replicate.com/jagilley/audio-ldm/badge)](https://replicate.com/jagilley/audio-ldm)

<!-- # [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) -->

**Generate speech, sound effects, music and beyond.**

This repo currently support: 

- **Text-to-Audio Generation**: Generate audio given text input.
- **Audio-to-Audio Generation**: Given an audio, generate another audio that contain the same type of sound. 
- **Text-guided Audio-to-Audio Style Transfer**: Transfer the sound of an audio into another one using the text description.

<hr>

## Important tricks to make your generated audio sound better
1. Try to provide more hints to AudioLDM, such as using more adjectives to describe your sound (e.g., clearly, high quality) or make your target more specific (e.g., "water stream in a forest" instead of "stream"). This can make sure AudioLDM understand what you want. 
2. Try to use different random seeds, which can affect the generation quality significantly sometimes.
3. It's best to use general terms like 'man' or 'woman' instead of specific names for individuals or abstract objects that humans may not be familiar with.

# Change Log

**2023-04-10**: Try to finetune AudioLDM with MusicCaps and AudioCaps datasets. Add three more checkpoints, including audioldm-m-text-ft, audioldm-s-text-ft, and audioldm-m-full.

**2023-03-04**: Add two more checkpoints, one is small model with more training steps, another is a large model. Add model selection in the Gradio APP.

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

:gear: How to choose between different model checkpoints?
```
# Add the --model_name parameter, choice={audioldm-m-text-ft, audioldm-s-text-ft, audioldm-m-full, audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}
audioldm --model_name audioldm-s-full
```

- :star: audioldm-m-full (default, **recommend**): the medium AudioLDM without finetuning and trained with audio embeddings as condition *(added 2023-04-10)*.
- :star: audioldm-s-full (**recommend**): the original open-sourced version *(added 2023-02-01)*.
- :star: audioldm-s-full-v2 (**recommend**): more training steps comparing with audioldm-s-full *(added 2023-03-04)*.
- audioldm-s-text-ft: the small AudioLDM finetuned with AudioCaps and MusicCaps audio-text pairs *(added 2023-04-10)*.
- audioldm-m-text-ft: the medium large AudioLDM finetuned with AudioCaps and MusicCaps audio-text pairs *(added 2023-04-10)*.
- audioldm-l-full: larger model comparing with audioldm-s-full *(added 2023-03-04)*.

> @haoheliu personally did a evaluation regarding the overall quality of the checkpoint, which gives audioldm-m-full (6.85/10), audioldm-s-full (6.62/10), audioldm-s-text-ft (6/10), audioldm-m-text-ft (5.46/10). These score are only for reference and may not reflect the true performance of the checkpoint. Checkpoint performance also varying with different text input as well.

:grey_question: For more options on guidance scale, batchsize, seed, ddim steps, etc., please run
```shell
audioldm -h
```
```console
usage: audioldm [-h] [--mode {generation,transfer}] [-t TEXT] [-f FILE_PATH] [--transfer_strength TRANSFER_STRENGTH] [-s SAVE_PATH] [--model_name {audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}] [-ckpt CKPT_PATH]
                [-b BATCHSIZE] [--ddim_steps DDIM_STEPS] [-gs GUIDANCE_SCALE] [-dur DURATION] [-n N_CANDIDATE_GEN_PER_TEXT] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  --mode {generation,transfer}
                        generation: text-to-audio generation; transfer: style transfer
  -t TEXT, --text TEXT  Text prompt to the model for audio generation, DEFAULT ""
  -f FILE_PATH, --file_path FILE_PATH
                        (--mode transfer): Original audio file for style transfer; Or (--mode generation): the guidance audio file for generating simialr audio, DEFAULT None
  --transfer_strength TRANSFER_STRENGTH
                        A value between 0 and 1. 0 means original audio without transfer, 1 means completely transfer to the audio indicated by text, DEFAULT 0.5
  -s SAVE_PATH, --save_path SAVE_PATH
                        The path to save model output, DEFAULT "./output"
  --model_name {audioldm-s-full,audioldm-l-full,audioldm-s-full-v2}
                        The checkpoint you gonna use, DEFAULT "audioldm-s-full"
  -ckpt CKPT_PATH, --ckpt_path CKPT_PATH
                        (deprecated) The path to the pretrained .ckpt model, DEFAULT None
  -b BATCHSIZE, --batchsize BATCHSIZE
                        Generate how many samples at the same time, DEFAULT 1
  --ddim_steps DDIM_STEPS
                        The sampling step for DDIM, DEFAULT 200
  -gs GUIDANCE_SCALE, --guidance_scale GUIDANCE_SCALE
                        Guidance scale (Large => better quality and relavancy to text; Small => better diversity), DEFAULT 2.5
  -dur DURATION, --duration DURATION
                        The duration of the samples, DEFAULT 10
  -n N_CANDIDATE_GEN_PER_TEXT, --n_candidate_gen_per_text N_CANDIDATE_GEN_PER_TEXT
                        Automatic quality control. This number control the number of candidates (e.g., generate three audios and choose the best to show you). A Larger value usually lead to better quality with heavier computation, DEFAULT 3
  --seed SEED           Change this value (any integer number) will lead to a different generation result. DEFAULT 42
```

For the evaluation of audio generative model, please refer to [audioldm_eval](https://github.com/haoheliu/audioldm_eval).

# Hugging Face 🧨 Diffusers

AudioLDM is available in the Hugging Face [🧨 Diffusers](https://github.com/huggingface/diffusers) library from v0.15.0 onwards. The official checkpoints can be found on the [Hugging Face Hub](https://huggingface.co/cvssp), alongside [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm) and [examples scripts](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm).

To install Diffusers and Transformers, run:
```bash
pip install --upgrade diffusers transformers
```

You can then load pre-trained weights into the [AudioLDM pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/audioldm) and generate text-conditional audio outputs:
```python
from diffusers import AudioLDMPipeline
import torch

repo_id = "cvssp/audioldm-s-full-v2"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]
```

# Web Demo

Integrated into [Hugging Face Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/haoheliu/audioldm-text-to-audio-generation)

# TuneFlow Demo

Try out AudioLDM as a [TuneFlow](https://tuneflow.com) plugin [![TuneFlow x AudioLDM](https://img.shields.io/badge/TuneFlow-AudioLDM-%23C563E6%20)](https://github.com/tuneflow/AudioLDM). See how it can work in a real DAW (Digital Audio Workstation). 

# TODO

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/haoheliuP)

- [x] Update the checkpoint with more training steps.
- [x] Update the checkpoint with more parameters (audioldm-l).
- [ ] Add AudioCaps finetuned AudioLDM-S model
- [x] Build pip installable package for commandline use
- [x] Build Gradio web application
- [ ] Add super-resolution, inpainting into Gradio web application
- [ ] Add style-transfer into Gradio web application
- [x] Add text-guided style transfer
- [x] Add audio-to-audio generation
- [x] Add audio super-resolution
- [x] Add audio inpainting

## Cite this work

If you found this tool useful, please consider citing
```bibtex
@article{liu2023audioldm,
  title={{AudioLDM}: Text-to-Audio Generation with Latent Diffusion Models},
  author={Liu, Haohe and Chen, Zehua and Yuan, Yi and Mei, Xinhao and Liu, Xubo and Mandic, Danilo and Wang, Wenwu and Plumbley, Mark D},
  journal={Proceedings of the International Conference on Machine Learning},
  year={2023}
  pages={21450-21474}
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
