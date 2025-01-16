import torch
import numpy as np
import soundfile as sf
from typing import Dict, Any
import io

class AudioPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.sample_rate = config['sample_rate']
        self.max_length = config['max_audio_length']

    def process(self, audio_bytes: bytes) -> torch.Tensor:
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_io)

        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        if sr != self.sample_rate:
            # use resampy or librosa
            pass

        max_samples = self.sample_rate * self.max_length
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return torch.from_numpy(audio)
