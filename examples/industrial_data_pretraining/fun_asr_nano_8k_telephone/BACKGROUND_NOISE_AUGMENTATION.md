# Background Noise Augmentation for Telephone Channel Simulation

## Overview

This implementation adds support for background noise augmentation to the telephone channel simulation. The noise augmentation simulates the real-world scenario where a microphone picks up both the speaker's voice and background noise before the signal passes through the telephone channel.

## Implementation Details

### Key Features

1. **Noise Manifest File Support**: Reads noise files from a SCP format manifest file
   - Format: `utt_id /path/to/noise/audio`
   - Example: `/data/speech/open/data/noise/manifests/musan_noise/noise.scp`

2. **Random SNR Mixing**: Randomly selects SNR within a specified range for each sample
   - Configurable SNR range (e.g., 5-20 dB)
   - Different noise ratio for each training sample

3. **Correct Processing Order**: Noise is added BEFORE channel simulation and resampling
   - Real-world scenario: Microphone → (Voice + Background Noise) → Channel → Transmission
   - Implementation: Background Noise → Downsample → Bandpass Filter → Codec → Telephone Noise

### Two Usage Methods

#### Method 1: Offline Data Preparation (Recommended for fixed datasets)

Use `data_simulation.py` to prepare noisy data offline:

```bash
python data/data_simulation.py \
    --input /path/to/16khz/train.jsonl \
    --output /path/to/8khz/train_noisy.jsonl \
    --output_audio_dir /path/to/8khz/audio_noisy \
    --target_fs 8000 \
    --bg_noise_scp /data/speech/open/data/noise/manifests/musan_noise/noise.scp \
    --bg_noise_snr_min 5 \
    --bg_noise_snr_max 20
```

**Advantages:**
- Pre-generate noisy data once
- Faster training (no runtime noise mixing)
- Reproducible results

#### Method 2: Runtime Data Augmentation (Recommended for dynamic training)

Configure noise augmentation in the training configuration:

```yaml
# config_8k_telephone.yaml
dataset_conf:
    preprocessor_speech: SpeechPreprocessNoiseAug
    preprocessor_speech_conf:
        noise_scp: /data/speech/open/data/noise/manifests/musan_noise/noise.scp
        snr_range: [5, 20]
        noise_apply_prob: 0.8  # 80% probability
```

**Advantages:**
- Different noise for each epoch
- Better data diversity
- Saves storage space
- Dynamic parameter adjustment

## Noise Manifest File Format

The noise manifest file should be in SCP (Kaldi script) format:

```
noise-sound-0001 /path/to/noise/audio1.wav
noise-sound-0002 /path/to/noise/audio2.wav
noise-sound-0003 /path/to/noise/audio3.wav
```

Each line contains:
- Utterance ID (can be any unique identifier)
- Space separator
- Absolute path to the noise audio file

## Processing Pipeline

The complete processing pipeline for telephone channel simulation with background noise:

```
Original Audio (16kHz)
    ↓
[Add Background Noise]  ← Random noise from manifest with random SNR
    ↓
[Downsample to 8kHz]
    ↓
[Bandpass Filter 300-3400Hz]  ← Telephone line frequency response
    ↓
[G.711 Codec]  ← μ-law or A-law compression
    ↓
[Add Telephone Line Noise]  ← White noise + power line interference
    ↓
Final 8kHz Telephone Audio
```

## Example: Using MUSAN Noise Database

If you have the MUSAN noise database:

1. Create a noise manifest file:
```bash
# List all noise files in SCP format
find /data/musan/noise -name "*.wav" | awk '{printf "noise-%04d %s\n", NR, $0}' > noise.scp
```

2. Use with data simulation:
```bash
python data/data_simulation.py \
    --input train.jsonl \
    --output train_8k_noisy.jsonl \
    --output_audio_dir audio_8k_noisy \
    --bg_noise_scp noise.scp \
    --bg_noise_snr_min 5 \
    --bg_noise_snr_max 20
```

3. Or use with training config:
```yaml
preprocessor_speech_conf:
    noise_scp: /path/to/noise.scp
    snr_range: [5, 20]
    noise_apply_prob: 0.8
```

## Parameters

### data_simulation.py Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--bg_noise_scp` | Path to noise manifest file | `None` |
| `--bg_noise_snr_min` | Minimum SNR in dB | `5` |
| `--bg_noise_snr_max` | Maximum SNR in dB | `20` |

### SpeechPreprocessNoiseAug Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `noise_scp` | Path to noise manifest file | `None` |
| `snr_range` | List of [min_snr, max_snr] in dB | `[5, 20]` |
| `noise_apply_prob` | Probability of applying noise (0.0-1.0) | `1.0` |

## Testing

Verify the implementation:

```bash
# Test data simulation with background noise
python data/data_simulation.py \
    --input test.wav \
    --output test_noisy.wav \
    --bg_noise_scp noise.scp \
    --bg_noise_snr_min 10 \
    --bg_noise_snr_max 15

# Compare original and noisy output
# You should hear background noise mixed with the original speech
```

## Notes

- The noise augmentation is applied at the original sampling rate before downsampling
- If the noise file is shorter than the audio, it will be repeated
- If the noise file is longer, a random segment will be selected
- Noise files are automatically resampled if needed
- This matches the real-world scenario where background noise is present before the signal enters the telephone channel

## References

- Problem statement: 实现上述方案规划，以及当给出 /data/speech/open/data/noise/manifests/musan_noise/noise.scp 这样的噪声文件时（utt_id /path/to/noise/audio）支持模拟噪声和随机 ratio，发生在模拟信道和升采样之前（真实情况也是麦克风拾音人声和背景噪音、然后电话信道）
- MUSAN database: https://www.openslr.org/17/
