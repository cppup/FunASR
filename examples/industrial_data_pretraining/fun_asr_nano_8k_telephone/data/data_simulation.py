#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Data simulation tool for 8kHz telephone channel audio.
Converts high-quality 16kHz audio to 8kHz telephone channel simulated audio with:
- Downsampling to 8kHz
- 300Hz-3400Hz bandpass filter (telephone line frequency response)
- G.711 (μ-law/A-law) codec compression simulation
- Telephone line noise (white noise + 50/60Hz power line interference)
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


class TelephoneChannelSimulator:
    """Simulate telephone channel effects on audio."""
    
    def __init__(
        self,
        target_fs=8000,
        low_freq=300,
        high_freq=3400,
        codec_type="mu-law",
        snr_db_range=(15, 25),
        power_line_freq=50,
        add_noise=True,
        add_codec=True,
        bg_noise_scp=None,
        bg_noise_snr_range=(5, 20),
    ):
        """
        Initialize telephone channel simulator.
        
        Args:
            target_fs: Target sampling rate (default: 8000 Hz)
            low_freq: Low cutoff frequency for bandpass filter (default: 300 Hz)
            high_freq: High cutoff frequency for bandpass filter (default: 3400 Hz)
            codec_type: Codec type, "mu-law" or "a-law" (default: "mu-law")
            snr_db_range: SNR range in dB for telephone line noise (default: (15, 25))
            power_line_freq: Power line frequency in Hz, 50 or 60 (default: 50)
            add_noise: Whether to add telephone line noise (default: True)
            add_codec: Whether to apply codec compression (default: True)
            bg_noise_scp: Path to background noise SCP file (utt_id /path/to/noise/audio)
            bg_noise_snr_range: SNR range in dB for background noise (default: (5, 20))
        """
        self.target_fs = target_fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.codec_type = codec_type
        self.snr_db_range = snr_db_range
        self.power_line_freq = power_line_freq
        self.add_noise = add_noise
        self.add_codec = add_codec
        self.bg_noise_scp = bg_noise_scp
        self.bg_noise_snr_range = bg_noise_snr_range
        self.bg_noise_list = []
        
        if bg_noise_scp and os.path.exists(bg_noise_scp):
            self._load_noise_manifest()
        elif bg_noise_scp:
            print(f"Warning: Background noise SCP file not found: {bg_noise_scp}")
    
    def _load_noise_manifest(self):
        """Load background noise manifest file in SCP format."""
        try:
            with open(self.bg_noise_scp, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        utt_id, noise_path = parts
                        if os.path.exists(noise_path):
                            self.bg_noise_list.append(noise_path)
                        else:
                            print(f"Warning: Noise file not found: {noise_path}")
            
            print(f"Loaded {len(self.bg_noise_list)} background noise files from {self.bg_noise_scp}")
        except Exception as e:
            print(f"Error loading noise manifest: {e}")
            self.bg_noise_list = []
    
    def _load_noise_segment(self, noise_path, target_length, fs):
        """Load and prepare background noise segment."""
        try:
            # Load noise audio
            noise, noise_fs = sf.read(noise_path)
            
            # Convert to mono if needed
            if len(noise.shape) > 1:
                noise = np.mean(noise, axis=1)
            
            # Resample if needed
            if noise_fs != fs:
                num_samples = int(len(noise) * fs / noise_fs)
                noise = signal.resample(noise, num_samples)
            
            # If noise is shorter than target, tile it
            if len(noise) < target_length:
                num_repeats = int(np.ceil(target_length / len(noise)))
                noise = np.tile(noise, num_repeats)[:target_length]
            
            # If noise is longer, randomly crop it
            if len(noise) > target_length:
                start_idx = np.random.randint(0, len(noise) - target_length + 1)
                noise = noise[start_idx:start_idx + target_length]
            
            return noise
        except Exception as e:
            print(f"Warning: Error loading noise file {noise_path}: {e}")
            return None
    
    def add_background_noise(self, audio, fs):
        """
        Add background noise to audio (simulates microphone picking up voice + background).
        This should be applied BEFORE channel simulation and resampling.
        
        Args:
            audio: Input audio signal
            fs: Sampling rate
        
        Returns:
            Audio with added background noise
        """
        if not self.bg_noise_list:
            return audio
        
        # Randomly select a noise file using random.choice for better performance
        noise_path = random.choice(self.bg_noise_list)
        
        # Load noise segment
        noise = self._load_noise_segment(noise_path, len(audio), fs)
        if noise is None:
            return audio
        
        # Calculate signal and noise power
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise ** 2)
        
        # Randomly select SNR within range
        snr_db = np.random.uniform(self.bg_noise_snr_range[0], self.bg_noise_snr_range[1])
        
        # Calculate scaling factor for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        # Scale and add noise
        scaled_noise = noise * scale
        noisy_audio = audio + scaled_noise
        
        return noisy_audio
        
    def resample_audio(self, audio, orig_fs):
        """Resample audio to target sampling rate."""
        if orig_fs == self.target_fs:
            return audio
        
        # Use polyphase filtering for high-quality resampling
        num_samples = int(len(audio) * self.target_fs / orig_fs)
        resampled = signal.resample(audio, num_samples)
        return resampled
    
    def bandpass_filter(self, audio, fs):
        """Apply bandpass filter to simulate telephone line frequency response."""
        # Normalize frequencies to Nyquist frequency
        nyquist = fs / 2
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, audio)
        return filtered
    
    def g711_mu_law_compress(self, audio, mu=255):
        """
        Apply G.711 μ-law compression.
        
        Args:
            audio: Input audio signal (normalized to [-1, 1])
            mu: μ-law parameter (default: 255)
        
        Returns:
            Compressed and decompressed audio
        """
        # Normalize to [-1, 1]
        audio = np.clip(audio, -1, 1)
        
        # μ-law compression
        sign = np.sign(audio)
        audio_abs = np.abs(audio)
        compressed = sign * np.log(1 + mu * audio_abs) / np.log(1 + mu)
        
        # Quantize to 8-bit (256 levels)
        quantized = np.round((compressed + 1) * 127.5)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        # μ-law decompression
        normalized = quantized.astype(np.float32) / 127.5 - 1
        sign_norm = np.sign(normalized)
        decompressed = sign_norm * (np.power(1 + mu, np.abs(normalized)) - 1) / mu
        
        return decompressed
    
    def g711_a_law_compress(self, audio, A=87.6):
        """
        Apply G.711 A-law compression.
        
        Args:
            audio: Input audio signal (normalized to [-1, 1])
            A: A-law parameter (default: 87.6)
        
        Returns:
            Compressed and decompressed audio
        """
        # Normalize to [-1, 1]
        audio = np.clip(audio, -1, 1)
        
        # A-law compression
        sign = np.sign(audio)
        audio_abs = np.abs(audio)
        
        compressed = np.zeros_like(audio)
        mask = audio_abs < (1 / A)
        compressed[mask] = sign[mask] * A * audio_abs[mask] / (1 + np.log(A))
        compressed[~mask] = sign[~mask] * (1 + np.log(A * audio_abs[~mask])) / (1 + np.log(A))
        
        # Quantize to 8-bit (256 levels)
        quantized = np.round((compressed + 1) * 127.5)
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)
        
        # A-law decompression
        normalized = quantized.astype(np.float32) / 127.5 - 1
        sign_norm = np.sign(normalized)
        decompressed = np.zeros_like(normalized)
        
        threshold = 1 / (1 + np.log(A))
        mask = np.abs(normalized) < threshold
        decompressed[mask] = sign_norm[mask] * np.abs(normalized[mask]) * (1 + np.log(A)) / A
        decompressed[~mask] = sign_norm[~mask] * (np.exp(np.abs(normalized[~mask]) * (1 + np.log(A)) - 1) - 1) / A
        
        return decompressed
    
    def add_telephone_noise(self, audio, fs):
        """
        Add telephone line noise (white noise + power line interference).
        
        Args:
            audio: Input audio signal
            fs: Sampling rate
        
        Returns:
            Audio with added noise
        """
        # Calculate signal power
        signal_power = np.mean(audio ** 2)
        
        # Random SNR within specified range
        snr_db = np.random.uniform(self.snr_db_range[0], self.snr_db_range[1])
        
        # Calculate noise power
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate white noise
        white_noise = np.random.randn(len(audio)) * np.sqrt(noise_power * 0.8)
        
        # Generate power line interference (50/60 Hz hum)
        t = np.arange(len(audio)) / fs
        power_line_noise = np.sqrt(noise_power * 0.2) * np.sin(2 * np.pi * self.power_line_freq * t)
        
        # Add harmonics of power line frequency
        power_line_noise += 0.3 * np.sqrt(noise_power * 0.2) * np.sin(2 * np.pi * 2 * self.power_line_freq * t)
        
        # Combine signal with noise
        noisy_audio = audio + white_noise + power_line_noise
        
        return noisy_audio
    
    def simulate(self, audio, orig_fs):
        """
        Apply complete telephone channel simulation.
        
        The order of operations mimics real-world scenario:
        1. Add background noise (microphone picks up voice + background)
        2. Resample to 8kHz
        3. Apply channel effects (bandpass filter, codec, telephone line noise)
        
        Args:
            audio: Input audio signal
            orig_fs: Original sampling rate
        
        Returns:
            Simulated telephone channel audio at target sampling rate
        """
        # Step 0: Add background noise BEFORE channel simulation (real-world order)
        # This simulates microphone picking up both voice and background noise
        if self.bg_noise_list:
            audio = self.add_background_noise(audio, orig_fs)
        
        # Step 1: Resample to 8kHz
        audio = self.resample_audio(audio, orig_fs)
        
        # Step 2: Apply bandpass filter (300-3400 Hz)
        audio = self.bandpass_filter(audio, self.target_fs)
        
        # Step 3: Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        # Step 4: Apply G.711 codec compression (optional)
        if self.add_codec:
            if self.codec_type == "mu-law":
                audio = self.g711_mu_law_compress(audio)
            elif self.codec_type == "a-law":
                audio = self.g711_a_law_compress(audio)
        
        # Step 5: Add telephone line noise (optional)
        if self.add_noise:
            audio = self.add_telephone_noise(audio, self.target_fs)
        
        # Step 6: Final normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        return audio


def process_audio_file(input_path, output_path, simulator):
    """Process a single audio file."""
    try:
        # Load audio
        audio, orig_fs = sf.read(input_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Simulate telephone channel
        simulated = simulator.simulate(audio, orig_fs)
        
        # Save simulated audio
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, simulated, simulator.target_fs)
        
        return True, None
    except Exception as e:
        return False, str(e)


def process_jsonl(input_jsonl, output_jsonl, output_audio_dir, simulator, audio_key="source"):
    """
    Process audio files listed in JSONL format.
    
    Args:
        input_jsonl: Input JSONL file path
        output_jsonl: Output JSONL file path
        output_audio_dir: Output directory for simulated audio files
        simulator: TelephoneChannelSimulator instance
        audio_key: Key in JSONL for audio file path (default: "source")
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_jsonl, 'w', encoding='utf-8') as fout:
        
        for line_no, line in enumerate(fin, 1):
            try:
                data = json.loads(line.strip())
                
                # Get input audio path
                if audio_key not in data:
                    print(f"Line {line_no}: Missing key '{audio_key}', skipping")
                    fail_count += 1
                    continue
                
                input_path = data[audio_key]
                
                # Generate output path
                input_filename = os.path.basename(input_path)
                name, ext = os.path.splitext(input_filename)
                output_filename = f"{name}_8k{ext}"
                output_path = os.path.join(output_audio_dir, output_filename)
                
                # Process audio
                success, error = process_audio_file(input_path, output_path, simulator)
                
                if success:
                    # Update data with new audio path
                    data[audio_key] = output_path
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
                    success_count += 1
                    
                    if success_count % 100 == 0:
                        print(f"Processed {success_count} files...")
                else:
                    print(f"Line {line_no}: Failed to process {input_path}: {error}")
                    fail_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: JSON decode error: {e}")
                fail_count += 1
            except Exception as e:
                print(f"Line {line_no}: Unexpected error: {e}")
                fail_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Output JSONL: {output_jsonl}")
    print(f"  Output audio directory: {output_audio_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate 8kHz telephone channel audio from high-quality audio"
    )
    
    # Input/output
    parser.add_argument('--input', type=str, required=True,
                        help='Input audio file or JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output audio file or JSONL file')
    parser.add_argument('--output_audio_dir', type=str, default=None,
                        help='Output directory for audio files (for JSONL mode)')
    parser.add_argument('--audio_key', type=str, default='source',
                        help='Key in JSONL for audio file path (default: source)')
    
    # Simulation parameters
    parser.add_argument('--target_fs', type=int, default=8000,
                        help='Target sampling rate (default: 8000)')
    parser.add_argument('--low_freq', type=int, default=300,
                        help='Low cutoff frequency for bandpass filter (default: 300)')
    parser.add_argument('--high_freq', type=int, default=3400,
                        help='High cutoff frequency for bandpass filter (default: 3400)')
    parser.add_argument('--codec_type', type=str, default='mu-law',
                        choices=['mu-law', 'a-law'],
                        help='Codec type (default: mu-law)')
    parser.add_argument('--snr_db_min', type=float, default=15,
                        help='Minimum SNR in dB for telephone line noise (default: 15)')
    parser.add_argument('--snr_db_max', type=float, default=25,
                        help='Maximum SNR in dB for telephone line noise (default: 25)')
    parser.add_argument('--power_line_freq', type=int, default=50,
                        choices=[50, 60],
                        help='Power line frequency in Hz (default: 50)')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable telephone line noise addition')
    parser.add_argument('--no_codec', action='store_true',
                        help='Disable codec compression')
    
    # Background noise parameters
    parser.add_argument('--bg_noise_scp', type=str, default=None,
                        help='Path to background noise SCP file (utt_id /path/to/noise/audio)')
    parser.add_argument('--bg_noise_snr_min', type=float, default=5,
                        help='Minimum SNR in dB for background noise (default: 5)')
    parser.add_argument('--bg_noise_snr_max', type=float, default=20,
                        help='Maximum SNR in dB for background noise (default: 20)')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = TelephoneChannelSimulator(
        target_fs=args.target_fs,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        codec_type=args.codec_type,
        snr_db_range=(args.snr_db_min, args.snr_db_max),
        power_line_freq=args.power_line_freq,
        add_noise=not args.no_noise,
        add_codec=not args.no_codec,
        bg_noise_scp=args.bg_noise_scp,
        bg_noise_snr_range=(args.bg_noise_snr_min, args.bg_noise_snr_max),
    )
    
    # Check if input is JSONL
    if args.input.endswith('.jsonl') or args.input.endswith('.json'):
        if args.output_audio_dir is None:
            print("Error: --output_audio_dir is required for JSONL mode")
            sys.exit(1)
        
        print(f"Processing JSONL: {args.input}")
        print(f"Simulation parameters:")
        print(f"  Target fs: {args.target_fs} Hz")
        print(f"  Bandpass: {args.low_freq}-{args.high_freq} Hz")
        print(f"  Codec: {args.codec_type} (enabled: {not args.no_codec})")
        print(f"  Telephone line noise: {args.snr_db_min}-{args.snr_db_max} dB SNR (enabled: {not args.no_noise})")
        print(f"  Power line: {args.power_line_freq} Hz")
        if args.bg_noise_scp:
            print(f"  Background noise: {args.bg_noise_scp}")
            print(f"  Background noise SNR: {args.bg_noise_snr_min}-{args.bg_noise_snr_max} dB")
        print()
        
        process_jsonl(
            args.input,
            args.output,
            args.output_audio_dir,
            simulator,
            audio_key=args.audio_key,
        )
    else:
        # Single file mode
        print(f"Processing audio file: {args.input}")
        success, error = process_audio_file(args.input, args.output, simulator)
        
        if success:
            print(f"Success: Output saved to {args.output}")
        else:
            print(f"Error: {error}")
            sys.exit(1)


if __name__ == "__main__":
    main()
