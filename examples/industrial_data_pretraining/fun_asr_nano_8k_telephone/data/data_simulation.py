#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Data simulation tool for 8kHz telephone channel audio.
Converts high-quality 16kHz audio to telephone channel simulated audio with:
- Downsampling to 8kHz
- 300Hz-3400Hz bandpass filter (telephone line frequency response)
- G.711 (μ-law/A-law) codec compression simulation
- Telephone line noise (white noise + 50/60Hz power line interference)
- Upsampling back to 16kHz (preserves telephone channel characteristics while
  maintaining compatibility with WavFrontend which requires 16kHz input)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import numpy as np
import soundfile as sf
from scipy import signal


class TelephoneChannelSimulator:
    """Simulate telephone channel effects on audio."""
    
    def __init__(
        self,
        target_fs=8000,
        output_fs=16000,
        low_freq=300,
        high_freq=3400,
        codec_type="mu-law",
        snr_db_range=(15, 25),
        power_line_freq=50,
        add_noise=True,
        add_codec=True,
    ):
        """
        Initialize telephone channel simulator.
        
        Args:
            target_fs: Target sampling rate for telephone simulation (default: 8000 Hz)
            output_fs: Final output sampling rate (default: 16000 Hz)
            low_freq: Low cutoff frequency for bandpass filter (default: 300 Hz)
            high_freq: High cutoff frequency for bandpass filter (default: 3400 Hz)
            codec_type: Codec type, "mu-law" or "a-law" (default: "mu-law")
            snr_db_range: SNR range in dB (default: (15, 25))
            power_line_freq: Power line frequency in Hz, 50 or 60 (default: 50)
            add_noise: Whether to add noise (default: True)
            add_codec: Whether to apply codec compression (default: True)
        """
        self.target_fs = target_fs
        self.output_fs = output_fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.codec_type = codec_type
        self.snr_db_range = snr_db_range
        self.power_line_freq = power_line_freq
        self.add_noise = add_noise
        self.add_codec = add_codec
        
    def resample_audio(self, audio, orig_fs, target_fs=None):
        """Resample audio to target sampling rate."""
        if target_fs is None:
            target_fs = self.target_fs
        if orig_fs == target_fs:
            return audio
        
        # Use polyphase filtering for high-quality resampling
        num_samples = int(len(audio) * target_fs / orig_fs)
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
        
        Args:
            audio: Input audio signal
            orig_fs: Original sampling rate
        
        Returns:
            Simulated telephone channel audio at output sampling rate (16kHz)
        """
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
        
        # Step 7: Upsample to output_fs (16kHz) to preserve telephone characteristics
        # while being compatible with WavFrontend
        if self.output_fs != self.target_fs:
            audio = self.resample_audio(audio, self.target_fs, target_fs=self.output_fs)
        
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
        sf.write(output_path, simulated, simulator.output_fs)
        
        return True, None
    except Exception as e:
        return False, str(e)


def _process_audio_worker(args):
    """Worker function for parallel processing of audio files.
    
    Args:
        args: Tuple of (input_path, output_path, simulator_params)
    
    Returns:
        Tuple of (line_no, input_path, output_path, success, error)
    """
    line_no, input_path, output_path, simulator_params = args
    
    try:
        # Recreate simulator in worker process
        simulator = TelephoneChannelSimulator(**simulator_params)
        success, error = process_audio_file(input_path, output_path, simulator)
        return (line_no, input_path, output_path, success, error)
    except Exception as e:
        return (line_no, input_path, output_path, False, str(e))


def process_jsonl(input_jsonl, output_jsonl, output_audio_dir, simulator, audio_key="source", num_workers=64):
    """
    Process audio files listed in JSONL format (with optional parallel processing).
    
    Args:
        input_jsonl: Input JSONL file path
        output_jsonl: Output JSONL file path
        output_audio_dir: Output directory for simulated audio files
        simulator: TelephoneChannelSimulator instance
        audio_key: Key in JSONL for audio file path (default: "source")
        num_workers: Number of parallel workers (default: 64, set to 1 for serial processing)
    """
    os.makedirs(output_audio_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    # Prepare simulator parameters for worker processes
    simulator_params = {
        'target_fs': simulator.target_fs,
        'output_fs': simulator.output_fs,
        'low_freq': simulator.low_freq,
        'high_freq': simulator.high_freq,
        'codec_type': simulator.codec_type,
        'snr_db_range': simulator.snr_db_range,
        'power_line_freq': simulator.power_line_freq,
        'add_noise': simulator.add_noise,
        'add_codec': simulator.add_codec,
    }
    
    # Read input JSONL and prepare tasks
    tasks = []
    jsonl_data = []
    
    with open(input_jsonl, 'r', encoding='utf-8') as fin:
        for line_no, line in enumerate(fin, 1):
            try:
                data = json.loads(line.strip())
                jsonl_data.append((line_no, data))
                
                # Get input audio path
                if audio_key not in data:
                    print(f"Line {line_no}: Missing key '{audio_key}', skipping")
                    continue
                
                input_path = data[audio_key]
                
                # Generate output path
                input_filename = os.path.basename(input_path)
                name, ext = os.path.splitext(input_filename)
                output_filename = f"{name}_8k{ext}"
                output_path = os.path.join(output_audio_dir, output_filename)
                
                tasks.append((line_no, input_path, output_path, simulator_params))
                
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: JSON decode error: {e}")
    
    success_count = 0
    fail_count = 0
    results = {}  # Store results by line_no
    
    # Process tasks in parallel
    if num_workers > 1:
        print(f"Processing {len(tasks)} audio files with {num_workers} workers...")
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_audio_worker, task): task[0] for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                try:
                    line_no, input_path, output_path, success, error = future.result()
                    results[line_no] = (input_path, output_path, success, error)
                    
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                        print(f"Line {line_no}: Failed to process {input_path}: {error}")
                    
                    completed += 1
                    if completed % max(1, len(tasks) // 10) == 0:
                        print(f"Processed {completed}/{len(tasks)} files...")
                        
                except Exception as e:
                    print(f"Worker error: {e}")
                    fail_count += 1
    else:
        # Serial processing
        print(f"Processing {len(tasks)} audio files serially...")
        for line_no, input_path, output_path, sim_params in tasks:
            simulator_instance = TelephoneChannelSimulator(**sim_params)
            success, error = process_audio_file(input_path, output_path, simulator_instance)
            results[line_no] = (input_path, output_path, success, error)
            
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Line {line_no}: Failed to process {input_path}: {error}")
            
            if (success_count + fail_count) % 100 == 0:
                print(f"Processed {success_count + fail_count} files...")
    
    # Write output JSONL with updated paths
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for line_no, data in jsonl_data:
            if line_no in results:
                input_path, output_path, success, error = results[line_no]
                if success:
                    # Update data with new audio path
                    data[audio_key] = output_path
                    fout.write(json.dumps(data, ensure_ascii=False) + '\n')
    
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
                        help='Target sampling rate for telephone simulation (default: 8000)')
    parser.add_argument('--output_fs', type=int, default=16000,
                        help='Final output sampling rate (default: 16000)')
    parser.add_argument('--low_freq', type=int, default=300,
                        help='Low cutoff frequency for bandpass filter (default: 300)')
    parser.add_argument('--high_freq', type=int, default=3400,
                        help='High cutoff frequency for bandpass filter (default: 3400)')
    parser.add_argument('--codec_type', type=str, default='mu-law',
                        choices=['mu-law', 'a-law'],
                        help='Codec type (default: mu-law)')
    parser.add_argument('--snr_db_min', type=float, default=15,
                        help='Minimum SNR in dB (default: 15)')
    parser.add_argument('--snr_db_max', type=float, default=25,
                        help='Maximum SNR in dB (default: 25)')
    parser.add_argument('--power_line_freq', type=int, default=50,
                        choices=[50, 60],
                        help='Power line frequency in Hz (default: 50)')
    parser.add_argument('--no_noise', action='store_true',
                        help='Disable noise addition')
    parser.add_argument('--no_codec', action='store_true',
                        help='Disable codec compression')
    
    # Parallel processing
    parser.add_argument('--num_workers', type=int, default=64,
                        help='Number of parallel workers for JSONL processing (default: 64, set to 1 for serial)')
    
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = TelephoneChannelSimulator(
        target_fs=args.target_fs,
        output_fs=args.output_fs,
        low_freq=args.low_freq,
        high_freq=args.high_freq,
        codec_type=args.codec_type,
        snr_db_range=(args.snr_db_min, args.snr_db_max),
        power_line_freq=args.power_line_freq,
        add_noise=not args.no_noise,
        add_codec=not args.no_codec,
    )
    
    # Check if input is JSONL
    if args.input.endswith('.jsonl') or args.input.endswith('.json'):
        if args.output_audio_dir is None:
            print("Error: --output_audio_dir is required for JSONL mode")
            sys.exit(1)
        
        print(f"Processing JSONL: {args.input}")
        print(f"Simulation parameters:")
        print(f"  Telephone simulation fs: {args.target_fs} Hz")
        print(f"  Output fs: {args.output_fs} Hz")
        print(f"  Bandpass: {args.low_freq}-{args.high_freq} Hz")
        print(f"  Codec: {args.codec_type} (enabled: {not args.no_codec})")
        print(f"  Noise: {args.snr_db_min}-{args.snr_db_max} dB SNR (enabled: {not args.no_noise})")
        print(f"  Power line: {args.power_line_freq} Hz")
        print()
        
        process_jsonl(
            args.input,
            args.output,
            args.output_audio_dir,
            simulator,
            audio_key=args.audio_key,
            num_workers=args.num_workers,
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
