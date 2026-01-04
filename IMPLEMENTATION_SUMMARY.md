# Implementation Summary: Background Noise Augmentation

## Problem Statement (Chinese)
实现上述方案规划，以及当给出 /data/speech/open/data/noise/manifests/musan_noise/noise.scp 这样的噪声文件时（utt_id /path/to/noise/audio）支持模拟噪声和随机 ratio，发生在模拟信道和升采样之前（真实情况也是麦克风拾音人声和背景噪音、然后电话信道）

## Translation
Implement the plan and support simulating noise with random ratio when given a noise file like /data/speech/open/data/noise/manifests/musan_noise/noise.scp (format: utt_id /path/to/noise/audio), occurring before channel simulation and upsampling (matching real-world scenario: microphone picks up voice + background noise, then telephone channel).

## Implementation Completed

### 1. New Preprocessor Class: `SpeechPreprocessNoiseAug`

**Location**: `funasr/datasets/audio_datasets/preprocessor.py`

**Features**:
- Loads noise files from SCP format manifest (utt_id /path/to/noise/audio)
- Random SNR selection within configurable range
- Probability-based noise application
- Automatic noise segment handling (repeat/crop/resample)
- Works at runtime during training

**Usage in training config**:
```yaml
dataset_conf:
    preprocessor_speech: SpeechPreprocessNoiseAug
    preprocessor_speech_conf:
        noise_scp: /data/speech/open/data/noise/manifests/musan_noise/noise.scp
        snr_range: [5, 20]
        noise_apply_prob: 0.8
```

### 2. Enhanced Data Simulation Script

**Location**: `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py`

**Features**:
- Adds background noise BEFORE channel simulation and downsampling
- Supports `--bg_noise_scp` parameter for noise manifest file
- Configurable SNR range via `--bg_noise_snr_min` and `--bg_noise_snr_max`
- Matches real-world processing order

**Usage for offline data preparation**:
```bash
python data/data_simulation.py \
    --input train.jsonl \
    --output train_8k.jsonl \
    --output_audio_dir audio_8k \
    --bg_noise_scp /data/speech/open/data/noise/manifests/musan_noise/noise.scp \
    --bg_noise_snr_min 5 \
    --bg_noise_snr_max 20
```

### 3. Processing Order (Correct Real-World Scenario)

```
Original Audio (16kHz)
    ↓
[Add Background Noise] ← Step 0: Random noise with random SNR
    ↓
[Downsample to 8kHz]   ← Step 1
    ↓
[Bandpass Filter 300-3400Hz] ← Step 2: Telephone line frequency response
    ↓
[G.711 Codec]          ← Step 3: μ-law or A-law compression
    ↓
[Add Telephone Line Noise]   ← Step 4: White noise + power line interference
    ↓
Final 8kHz Telephone Audio
```

This matches the real-world scenario:
1. **Microphone**: Picks up voice + background noise
2. **Channel**: Signal passes through telephone channel (bandpass, codec)
3. **Transmission**: Additional line noise is added

### 4. Documentation Created

1. **README_zh.md**: Updated with new parameters and usage examples
2. **IMPLEMENTATION_PLAN.md**: Updated with implementation details and design rationale
3. **BACKGROUND_NOISE_AUGMENTATION.md**: Comprehensive guide on using the feature
4. **config_with_noise_aug.yaml**: Example configuration snippet

### 5. Code Quality

- ✅ All Python files compile successfully
- ✅ Code review feedback addressed (memory efficiency, edge cases, error messages)
- ✅ Security scan completed (0 vulnerabilities found)
- ✅ Follows existing codebase conventions

## Key Design Decisions

1. **Two Usage Methods**:
   - **Method 1 (Offline)**: Pre-generate noisy data using `data_simulation.py`
     - Pros: Faster training, reproducible
     - Cons: Requires storage space
   - **Method 2 (Runtime)**: Apply noise during training using preprocessor
     - Pros: Data diversity, space-efficient
     - Cons: Slightly slower training

2. **Noise Processing Order**: Background noise is added BEFORE channel simulation and downsampling, matching the real-world scenario where microphone captures both voice and ambient noise before signal transmission.

3. **Random SNR**: Each sample gets a random SNR within the configured range, increasing training data diversity.

4. **Noise Segment Handling**:
   - If noise is shorter than audio: tile and crop
   - If noise is longer than audio: random crop
   - Automatic resampling to match audio sample rate

## Testing

- ✅ Python syntax verification with `py_compile`
- ✅ Code compiles without errors
- ✅ Security scan passed (CodeQL)
- ✅ Code review passed with all feedback addressed

## Files Modified

1. `funasr/datasets/audio_datasets/preprocessor.py` - Added `SpeechPreprocessNoiseAug` class
2. `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py` - Enhanced with background noise support
3. `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/README_zh.md` - Updated documentation
4. `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/IMPLEMENTATION_PLAN.md` - Updated with implementation details

## Files Created

1. `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/conf/config_with_noise_aug.yaml` - Example configuration
2. `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/BACKGROUND_NOISE_AUGMENTATION.md` - Comprehensive guide

## Example: Using MUSAN Noise Database

```bash
# Create noise manifest
find /data/musan/noise -name "*.wav" | \
    awk '{printf "noise-%04d %s\n", NR, $0}' > noise.scp

# Method 1: Offline preparation
python data/data_simulation.py \
    --input train.jsonl \
    --output train_8k_noisy.jsonl \
    --output_audio_dir audio_8k_noisy \
    --bg_noise_scp noise.scp \
    --bg_noise_snr_min 5 \
    --bg_noise_snr_max 20

# Method 2: Runtime augmentation (add to config)
# preprocessor_speech_conf:
#     noise_scp: /path/to/noise.scp
#     snr_range: [5, 20]
#     noise_apply_prob: 0.8
```

## Conclusion

The implementation fully satisfies the problem statement requirements:
- ✅ Supports noise manifest files in SCP format (utt_id /path/to/noise/audio)
- ✅ Implements random SNR ratio
- ✅ Applies noise BEFORE channel simulation and upsampling
- ✅ Matches real-world scenario (microphone → voice + background → channel)
- ✅ Provides both offline and runtime usage options
- ✅ Includes comprehensive documentation
- ✅ Passes all quality and security checks
