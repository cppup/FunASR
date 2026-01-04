# Fun-ASR-Nano 8kHz Telephone Channel Fine-tuning Solution

English | [简体中文](README_zh.md)

## Introduction

This solution provides a specialized fine-tuning toolkit for Fun-ASR-Nano model targeting 8kHz narrowband telephone channel scenarios (customer service centers, telemarketing, VoIP communications, etc.). Compared to standard 16kHz wideband audio, 8kHz narrowband audio faces the following challenges:

- **Spectrum Truncation**: Nyquist frequency limited to 4kHz, high-frequency components lost
- **Codec Distortion**: Quantization noise from G.711 (μ-law/A-law) companding
- **Channel Noise**: Line noise, power line hum (50/60Hz) interference
- **Conversational Features**: Disfluencies, interruptions, informal expressions

This solution helps users quickly achieve high-accuracy speech recognition for 8kHz telephone scenarios through data simulation, targeted training configuration, and inference tools.

## Features

- ✅ **Data Simulation Tool**: Generate 8kHz telephone channel simulated data from 16kHz high-quality audio
- ✅ **Optimized Training Configuration**: Model configuration optimized for 8kHz narrowband signals
- ✅ **Distributed Training**: Support for multi-GPU and DeepSpeed accelerated training
- ✅ **Inference & Evaluation**: Support single file, batch, and JSONL evaluation with CER/WER metrics
- ✅ **Hotword Enhancement**: Support hotword-biased decoding for improved domain terminology recognition

## Directory Structure

```
fun_asr_nano_8k_telephone/
├── README.md                       # English documentation (this file)
├── README_zh.md                    # Chinese documentation
├── requirements.txt                # Python dependencies
├── conf/
│   └── config_8k_telephone.yaml    # 8kHz training configuration
├── data/
│   └── data_simulation.py          # 16kHz→8kHz data simulation tool
├── finetune_8k_telephone.sh        # Fine-tuning training script
└── inference_8k.py                 # Inference script
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Data Preparation

#### 1.1 Prepare 16kHz High-Quality Data

First, prepare training data in JSONL format with 16kHz high-quality audio:

```jsonl
{"source": "/path/to/audio1.wav", "target": "This is the first audio transcription"}
{"source": "/path/to/audio2.wav", "target": "This is the second audio transcription"}
```

#### 1.2 Generate 8kHz Telephone Channel Simulated Data

Use the data simulation tool to convert 16kHz audio to 8kHz telephone channel simulated data:

```bash
python data/data_simulation.py \
    --input /path/to/16khz/train.jsonl \
    --output /path/to/8khz/train.jsonl \
    --output_audio_dir /path/to/8khz/audio \
    --audio_key source \
    --target_fs 8000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 15 \
    --snr_db_max 25 \
    --power_line_freq 50
```

**Parameter Description:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--input` | Input JSONL file path | Required |
| `--output` | Output JSONL file path | Required |
| `--output_audio_dir` | Output audio directory | Required (JSONL mode) |
| `--audio_key` | Key for audio path in JSONL | `source` |
| `--target_fs` | Target sampling rate (Hz) | `8000` |
| `--low_freq` | Bandpass filter low cutoff (Hz) | `300` |
| `--high_freq` | Bandpass filter high cutoff (Hz) | `3400` |
| `--codec_type` | Codec type (`mu-law` or `a-law`) | `mu-law` |
| `--snr_db_min` | Minimum SNR (dB) | `15` |
| `--snr_db_max` | Maximum SNR (dB) | `25` |
| `--power_line_freq` | Power line frequency (50 or 60 Hz) | `50` |
| `--no_noise` | Disable noise addition | `False` |
| `--no_codec` | Disable codec companding | `False` |

**Data Simulation Pipeline:**

1. Downsample to 8kHz
2. Apply 300-3400 Hz bandpass filter (simulates telephone line frequency response)
3. Apply G.711 μ-law/A-law codec companding
4. Add telephone line noise (white noise + 50/60Hz power line interference)

### 2. Model Fine-tuning

#### 2.1 Configure Training Script

Edit `finetune_8k_telephone.sh` and modify key parameters:

```bash
# Input data (simulated 8kHz data)
train_data="/path/to/8khz/train.jsonl"
val_data="/path/to/8khz/val.jsonl"

# GPU configuration
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Training hyperparameters
batch_size=4
max_epoch=20
learning_rate=0.00005  # 5e-5
```

#### 2.2 Start Training

```bash
# Start training directly (assuming data is ready)
bash finetune_8k_telephone.sh 1 1

# Full pipeline (data simulation + training)
bash finetune_8k_telephone.sh -1 1
```

**Training Stages:**

- **Stage -1**: Data simulation (16kHz → 8kHz telephone channel)
- **Stage 0**: Data preparation (verify data files, count samples)
- **Stage 1**: Fine-tuning training
- **Stage 2**: Evaluation

#### 2.3 Distributed Training

The script automatically supports multi-GPU distributed training. For larger-scale training, enable DeepSpeed:

```bash
# Modify finetune_8k_telephone.sh
use_deepspeed=true
deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage1.json"
```

### 3. Model Inference

#### 3.1 Single File Inference

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/test.wav \
    --output_dir ./output \
    --device cuda:0
```

#### 3.2 Batch Inference

```bash
# Prepare audio file list
echo "/path/to/audio1.wav" > audio_list.txt
echo "/path/to/audio2.wav" >> audio_list.txt

python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_list audio_list.txt \
    --output_dir ./output \
    --batch_size 4 \
    --device cuda:0
```

#### 3.3 JSONL Evaluation

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --test_data /path/to/test.jsonl \
    --output_dir ./output \
    --batch_size 4 \
    --audio_key source \
    --text_key target \
    --device cuda:0
```

Output includes:
- `results.jsonl`: Recognition results and CER/WER for each sample
- `metrics.json`: Overall evaluation metrics (average CER/WER, RTF, etc.)

#### 3.4 Hotword Enhancement

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/test.wav \
    --output_dir ./output \
    --hotwords "customer service,order number,refund" \
    --device cuda:0
```

#### 3.5 VAD Long Audio Segmentation

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/long_audio.wav \
    --output_dir ./output \
    --use_vad \
    --vad_model fsmn-vad \
    --max_single_segment_time 30000 \
    --device cuda:0
```

## Training Configuration

Configuration file is located at `conf/config_8k_telephone.yaml`. Key configurations:

### Frontend Configuration (8kHz Adaptation)

```yaml
frontend_conf:
    fs: 8000          # 8kHz sampling rate
    n_fft: 200        # 25ms window (200 samples @ 8kHz)
    hop_length: 80    # 10ms hop (80 samples @ 8kHz)
    n_mels: 80        # Number of mel filterbanks
```

### Audio Encoder Configuration (Partial Fine-tuning)

```yaml
audio_encoder_conf:
    freeze: false           # Allow fine-tuning
    freeze_layer_num: 12    # Freeze first 12 layers, fine-tune remaining
```

**Design Philosophy:**
- First 12 layers retain general feature extraction capability
- Remaining layers adapt to 8kHz narrowband signal characteristics

### LLM Configuration (LoRA Fine-tuning)

```yaml
llm_conf:
    freeze: true      # Freeze base LLM
    use_lora: true    # Enable LoRA efficient fine-tuning
    lora_conf:
        r: 32                 # LoRA rank
        lora_alpha: 64        # LoRA alpha scaling
        target_modules:       # Target modules
            - "q_proj"
            - "v_proj"
            - "k_proj"
            - "o_proj"
            - "gate_proj"
            - "up_proj"
            - "down_proj"
```

**LoRA Advantages:**
- Significantly reduce trainable parameters (typically < 1% of original model)
- Accelerate training, reduce memory usage
- Protect pre-trained knowledge, prevent catastrophic forgetting

### Audio Adaptor Configuration (Full Fine-tuning)

```yaml
audio_adaptor: Linear
audio_adaptor_conf:
    downsample_rate: 5
    llm_dim: 1536      # Qwen2.5-1.5B hidden dimension
    encoder_dim: 512   # Audio encoder output dimension
```

### SpecAug Configuration (8kHz Optimization)

```yaml
specaug_conf:
    freq_mask_width_range: [0, 15]  # Reduce frequency mask range (narrower 8kHz spectrum)
    time_mask_width_range: [0, 10]  # Moderately reduce time mask range
```

### Optimizer Configuration

```yaml
optim: adamw
optim_conf:
    lr: 0.00005         # 5e-5, lower learning rate to protect pre-trained knowledge
    weight_decay: 0.01
```

## Performance Optimization Tips

### 1. Data Augmentation

- Use different SNR ranges (10-30 dB) to generate diverse training data
- Mix μ-law and A-law codec simulations
- Add real telephone recordings as supplementary training data

### 2. Training Strategy

- **Warmup**: Use smaller learning rate for first 1000 steps
- **LR Scheduling**: Use warmup + cosine decay
- **Gradient Accumulation**: Increase `accum_grad` when memory limited (e.g., 2 or 4)

### 3. Hyperparameter Tuning

| Hyperparameter | Recommended Range | Notes |
|----------------|-------------------|-------|
| Learning Rate | 3e-5 ~ 1e-4 | Lower LR protects pre-trained knowledge |
| Batch Size | 2 ~ 8 | Adjust based on GPU memory |
| LoRA Rank | 16 ~ 64 | Higher = more expressive, but more expensive |
| Freeze Layers | 8 ~ 16 | More = retain more general features |

### 4. Memory Optimization

```bash
# Enable DeepSpeed ZeRO Stage 2
use_deepspeed=true
deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage2.json"

# Reduce batch size, increase gradient accumulation
batch_size=2
accum_grad=4
```

## FAQ

### Q1: How much training data is needed?

**Recommendation:**
- Minimum: 1000 hours of 16kHz high-quality data → simulate to 8kHz
- Recommended: 5000+ hours (covering various accents, noise scenarios)

### Q2: How to handle real telephone recordings?

Real telephone recordings are typically already 8kHz and can be used directly for fine-tuning. Recommendations:
1. Skip data simulation step
2. Ensure audio sampling rate is 8000 Hz
3. Mix real recordings with simulated data for training

### Q3: How long until convergence?

Depends on data size and GPU configuration:
- Small scale (1000 hours): 4-8 epochs, ~1-2 days (4x V100)
- Large scale (10000 hours): 10-15 epochs, ~5-7 days (8x A100)

### Q4: What are expected CER/WER metrics?

Reference metrics (depends on data quality and scenario):
- Clear telephone (SNR > 20dB): CER < 5%, WER < 10%
- Normal telephone (SNR 15-20dB): CER 8-12%, WER 15-20%
- Noisy telephone (SNR < 15dB): CER 15-25%, WER 25-35%

### Q5: How to further improve accuracy?

1. **Data Quality**:
   - Clean incorrect annotations
   - Add domain-relevant training data
   - Balance different accents and noise scenarios

2. **Model Optimization**:
   - Increase LoRA rank (e.g., 64)
   - Unfreeze more encoder layers
   - Increase training epochs

3. **Inference Optimization**:
   - Use hotword enhancement
   - Combine with language model rescoring
   - Post-processing rule optimization

## Technical Details

### Telephone Channel Simulation Principles

#### 1. Bandpass Filter

Telephone line frequency response is 300-3400 Hz, simulated using 4th-order Butterworth bandpass filter:

```python
low = 300 / (fs / 2)   # Normalized low frequency
high = 3400 / (fs / 2) # Normalized high frequency
b, a = signal.butter(4, [low, high], btype='band')
filtered = signal.filtfilt(b, a, audio)
```

#### 2. G.711 μ-law Companding

μ-law companding formula (μ = 255):

```
Compression: F(x) = sign(x) * log(1 + μ|x|) / log(1 + μ)
Expansion: F⁻¹(y) = sign(y) * (1/μ) * [(1 + μ)^|y| - 1]
```

Quantized to 8-bit (256 levels) to simulate G.711 codec.

#### 3. Noise Model

- **White Noise**: Gaussian white noise, 80% of total noise power
- **Power Line Interference**: 50/60 Hz sine wave and second harmonic, 20% of total noise power

```python
white_noise = np.random.randn(len(audio)) * sqrt(0.8 * noise_power)
power_line = sqrt(0.2 * noise_power) * sin(2π * 50 * t)
```

### LoRA Fine-tuning Principles

LoRA (Low-Rank Adaptation) achieves efficient fine-tuning through low-rank matrix decomposition:

```
W' = W + ΔW = W + BA
```

Where:
- W: Pre-trained weights (frozen)
- B: r × d matrix
- A: d × r matrix
- r: Rank (r << d)

**Parameter Count Comparison:**
- Original model: d × d
- LoRA: 2 × d × r (significantly less when r << d)

Example: d = 4096, r = 32
- Original model: 16,777,216 parameters
- LoRA: 262,144 parameters (98.4% reduction)

## Citation

If this solution helps your research or project, please cite:

```bibtex
@misc{funasr_8k_telephone,
    title={Fun-ASR-Nano 8kHz Telephone Channel Fine-tuning Scheme},
    author={FunASR Team},
    year={2025},
    howpublished={\url{https://github.com/modelscope/FunASR}},
}
```

## License

This project follows [MIT License](../../../LICENSE).

## Contact

- GitHub Issues: https://github.com/modelscope/FunASR/issues
- ModelScope: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

## Changelog

- **2025-01**: Initial release
  - Data simulation tool
  - 8kHz training configuration
  - Inference & evaluation script
  - Complete documentation
