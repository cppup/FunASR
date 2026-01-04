import os
import json
import torch
import logging
import concurrent.futures
import librosa
import torch.distributed as dist
from typing import Collection
import torch
import torchaudio
from torch import nn
import random
import re
import numpy as np
from funasr.tokenizer.cleaner import TextCleaner
from funasr.register import tables


@tables.register("preprocessor_classes", "SpeechPreprocessSpeedPerturb")
class SpeechPreprocessSpeedPerturb(nn.Module):
    def __init__(self, speed_perturb: list = None, **kwargs):
        super().__init__()
        self.speed_perturb = speed_perturb

    def forward(self, waveform, fs, **kwargs):
        if self.speed_perturb is None:
            return waveform
        speed = random.choice(self.speed_perturb)
        if speed != 1.0:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.tensor(waveform)
            waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.view(1, -1), fs, [["speed", str(speed)], ["rate", str(fs)]]
            )
            waveform = waveform.view(-1)

        return waveform


@tables.register("preprocessor_classes", "SpeechPreprocessNoiseAug")
class SpeechPreprocessNoiseAug(nn.Module):
    """
    Add background noise to speech signal with random SNR.
    This simulates real-world scenarios where microphone picks up both voice and background noise.
    """
    def __init__(
        self, 
        noise_scp: str = None,
        snr_range: list = None,
        noise_apply_prob: float = 1.0,
        **kwargs
    ):
        """
        Initialize noise augmentation preprocessor.
        
        Args:
            noise_scp: Path to noise manifest file in SCP format (utt_id /path/to/noise/audio)
            snr_range: List of two values [min_snr, max_snr] in dB. Default: [5, 20]
            noise_apply_prob: Probability of applying noise augmentation. Default: 1.0
        """
        super().__init__()
        self.noise_scp = noise_scp
        self.snr_range = snr_range if snr_range is not None else [5, 20]
        self.noise_apply_prob = noise_apply_prob
        self.noise_list = []
        
        if noise_scp and os.path.exists(noise_scp):
            self._load_noise_manifest()
        elif noise_scp:
            logging.warning(f"Noise SCP file not found: {noise_scp}, noise augmentation will be disabled")
    
    def _load_noise_manifest(self):
        """Load noise manifest file in SCP format."""
        try:
            with open(self.noise_scp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, noise_path = parts
                        if os.path.exists(noise_path):
                            self.noise_list.append(noise_path)
                        else:
                            logging.warning(f"Noise file not found: {noise_path}")
            
            logging.info(f"Loaded {len(self.noise_list)} noise files from {self.noise_scp}")
        except Exception as e:
            logging.error(f"Error loading noise manifest: {e}")
            self.noise_list = []
    
    def _load_noise(self, noise_path, target_length, fs):
        """Load and prepare noise segment."""
        try:
            # Load noise audio
            noise, noise_fs = torchaudio.load(noise_path)
            
            # Convert to mono if needed
            if noise.shape[0] > 1:
                noise = noise.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if noise_fs != fs:
                resampler = torchaudio.transforms.Resample(noise_fs, fs)
                noise = resampler(noise)
            
            noise = noise.squeeze()
            
            # If noise is shorter than target, repeat it
            if len(noise) < target_length:
                num_repeats = int(np.ceil(target_length / len(noise)))
                noise = noise.repeat(num_repeats)
            
            # If noise is longer, randomly crop it
            if len(noise) > target_length:
                start_idx = random.randint(0, len(noise) - target_length)
                noise = noise[start_idx:start_idx + target_length]
            
            return noise
        except Exception as e:
            logging.warning(f"Error loading noise file {noise_path}: {e}")
            return None
    
    def _add_noise(self, waveform, noise, snr_db):
        """Add noise to waveform with specified SNR."""
        # Calculate signal and noise power
        signal_power = torch.mean(waveform ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Calculate scaling factor for desired SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power_target = signal_power / (10 ** (SNR / 10))
        snr_linear = 10 ** (snr_db / 10)
        scale = torch.sqrt(signal_power / (noise_power * snr_linear))
        
        # Scale and add noise
        scaled_noise = noise * scale
        noisy_waveform = waveform + scaled_noise
        
        return noisy_waveform
    
    def forward(self, waveform, fs, **kwargs):
        """Apply noise augmentation to waveform."""
        if not self.noise_list or random.random() > self.noise_apply_prob:
            return waveform
        
        # Ensure waveform is a tensor
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Randomly select a noise file
        noise_path = random.choice(self.noise_list)
        
        # Load noise segment
        noise = self._load_noise(noise_path, len(waveform), fs)
        if noise is None:
            return waveform
        
        # Randomly select SNR
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        
        # Add noise to waveform
        noisy_waveform = self._add_noise(waveform, noise, snr_db)
        
        return noisy_waveform


@tables.register("preprocessor_classes", "TextPreprocessSegDict")
class TextPreprocessSegDict(nn.Module):
    def __init__(
        self,
        seg_dict: str = None,
        text_cleaner: Collection[str] = None,
        split_with_space: bool = False,
        **kwargs
    ):
        super().__init__()

        self.text_cleaner = TextCleaner(text_cleaner)

    def forward(self, text, **kwargs):
        text = self.text_cleaner(text)

        return text
