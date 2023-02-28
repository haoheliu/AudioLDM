from tuneflow_py import TuneflowPlugin, ParamDescriptor, Song, TrackType, WidgetType, LabelText
from typing import Dict, Any
from audioldm import text_to_audio, build_model
from pathlib import Path
import traceback
from io import BytesIO
import soundfile as sf
from typing import List


class AudioLDMPlugin(TuneflowPlugin):
    @staticmethod
    def provider_id() -> str:
        return 'andantei'

    @staticmethod
    def plugin_id() -> str:
        return 'audioldm-generate'

    @staticmethod
    def provider_display_name() -> LabelText:
        return {
            "zh": "Andantei行板",
            "en": "Andantei"
        }

    @staticmethod
    def params(song: Song) -> Dict[str, ParamDescriptor]:
        return {
            "prompt": {
                "displayName": {
                    "en": "Prompt",
                    "zh": "提示词"
                },
                "description": {
                    "en": "A short sentence to describe the audio you want to generate",
                    "zh": "用一段简短的文字描述你想要的音频"
                },
                "defaultValue": None,
                "widget": {
                    "type": WidgetType.TextArea.value,
                    "config": {
                        "placeholder": {
                            "zh": "样例：斧头正在伐木",
                            "en": "e.g. A hammer is hitting a tree"
                        },
                        "maxLength": 140
                    }
                }
            },
            "guidance_scale": {
                "displayName": {
                    "en": "Guidance Scale",
                    "zh": "提示强度"
                },
                "description": {
                    "en": "Larger value yields results more relavant to the prompt, smaller value yields more diversity",
                    "zh": "值越大，生成结果越贴近提示词，值越小，生成结果越发散"
                },
                "defaultValue": 2.5,
                "widget": {
                    "type": WidgetType.InputNumber.value,
                    "config": {
                        "minValue": 0.1,
                        "maxValue": 10,
                        "step": 0.1
                    }
                }
            },
            "duration": {
                "displayName": {
                    "en": "Duration (seconds)",
                    "zh": "长度 (秒)"
                },
                "defaultValue": 10,
                "widget": {
                    "type": WidgetType.InputNumber.value,
                    "config": {
                        "minValue": 2.5,
                        "maxValue": 10,
                        "step": 2.5
                    }
                }
            },
            "randomSeed": {
                "displayName": {
                    "en": "Random Seed",
                    "zh": "随机因子"
                },
                "defaultValue": 42,
                "description": {
                    "en": "Using the same params and random seed generates the same response",
                    "zh": "使用相同的参数值和随机因子可以生成相同的结果"
                },
                "widget": {
                    "type": WidgetType.InputNumber.value,
                    "config": {
                        "minValue": 1,
                        "maxValue": 99999999,
                        "step": 1
                    }
                }
            }
        }

    @staticmethod
    def run(song: Song, params: Dict[str, Any]):
        model_path = str(Path(__file__).parent.joinpath('ckpt/ldm_trimmed.ckpt').absolute())
        model = build_model(ckpt_path=model_path)
        # TODO: Support prompt i18n
        file_bytes_list = AudioLDMPlugin._text2audio(
            model,
            text=params["prompt"],
            duration=params["duration"],
            guidance_scale=params["guidance_scale"],
            # Randomize seed.
            random_seed=params["randomSeed"])
        for file_bytes in file_bytes_list:
            try:
                file_bytes.seek(0)
                track = song.create_track(type=TrackType.AUDIO_TRACK)
                track.create_audio_clip(clip_start_tick=0, audio_clip_data={
                    "audio_data": {
                        "format": "wav",
                        "data": file_bytes.read()
                    },
                    "duration": params["duration"],
                    "start_tick": 0
                })
            except:
                print(traceback.format_exc())

    @staticmethod
    def _text2audio(model, text, duration, guidance_scale, random_seed):
        # print(text, length, guidance_scale)
        waveform = text_to_audio(
            model,
            text=text,
            seed=random_seed,
            duration=duration,
            guidance_scale=guidance_scale,
            n_candidate_gen_per_text=3,
            batchsize=1,
        )
        return AudioLDMPlugin._save_wave(waveform)

    @staticmethod
    def _save_wave(waveform):
        saved_file_bytes: List[BytesIO] = []
        for i in range(waveform.shape[0]):
            file_bytes = BytesIO()
            sf.write(file_bytes, waveform[i, 0], samplerate=16000, format="wav")
            saved_file_bytes.append(file_bytes)
        return saved_file_bytes
