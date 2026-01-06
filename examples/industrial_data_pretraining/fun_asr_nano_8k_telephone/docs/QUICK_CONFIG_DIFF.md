# Quick Configuration Diff - Fine-tuning vs Pretrained Model

## Visual Comparison Format

```
Parameter: value
├─ Pretrained:        [value in pretrained model]
├─ config_8k_telephone: [value in main config]
├─ stage1_encoder_adapt: [value in stage1 config]
└─ Status: 🔵/🟡/🟠/⚪
```

---

## 1️⃣ AUDIO ENCODER - THE MOST CRITICAL CHANGE

### `audio_encoder_conf.freeze`
```
Pretrained:           true  (FROZEN for inference)
config_8k_telephone:  false (🔴 TRAINABLE)
stage1_encoder_adapt: false (🔴 TRAINABLE)
```
**Impact**: Encoder weights will be updated during fine-tuning. ~150M parameters become trainable.

### `frontend_conf.fs` (Sample Rate)
```
Pretrained:           16000 Hz
config_8k_telephone:  16000 Hz (8kHz audio upsampled to 16kHz)
stage1_encoder_adapt: 8000  Hz ⚠️ (Different from config_8k_telephone)
```
**Note**: stage1 uses 8kHz frontend; config_8k_telephone uses 16kHz. This is intentional for different training strategies.

---

## 2️⃣ CTC DECODER - STAGE DEPENDENT

### `ctc_decoder_conf.freeze`
```
Pretrained:           false (trainable for ASR)
config_8k_telephone:  false (TRAINABLE) 🔴
stage1_encoder_adapt: true  (FROZEN)    🟡
```
**Strategy**:
- **stage1**: Focus on encoder adaptation, freeze CTC decoder
- **config_8k_telephone**: Train both encoder and CTC decoder

---

## 3️⃣ LEARNING RATE - CRITICAL FOR CONVERGENCE

### `optim_conf.lr`
```
Pretrained:           5e-6  (0.000005 - very tiny for finetuning)
config_8k_telephone:  5e-5  (0.00005 - 10x higher) 🟡
stage1_encoder_adapt: 1e-4  (0.0001 - 20x higher) 🟡
```
**Why changed**: Pretrained LR is for full training with DeepSpeed. Fine-tuning needs higher LR.
**Why different by stage**: stage1 only trains encoder → higher LR acceptable.

---

## 4️⃣ TRAINING CONFIGURATION - ADAPTED FOR FINETUNING

### `train_conf.max_epoch`
```
Pretrained:           2    (Two epochs in original training)
config_8k_telephone:  20   (🟡 10x more epochs for finetuning)
stage1_encoder_adapt: 10   (🟡 5x more epochs)
```

### `train_conf.use_deepspeed`
```
Pretrained:           true  (Enabled for distributed training)
config_8k_telephone:  false (🟡 Disabled for single GPU fine-tuning)
stage1_encoder_adapt: false (🟡 Disabled)
```

### `train_conf.keep_nbest_models`
```
Pretrained:           200  (Keep all checkpoints)
config_8k_telephone:  10   (🟡 Keep only best 10)
stage1_encoder_adapt: 5    (🟡 Keep only best 5)
```
**Why reduced**: Save disk space during fine-tuning.

### `train_conf.log_interval`
```
Pretrained:           100
config_8k_telephone:  10   (🟡 More frequent logging)
stage1_encoder_adapt: 10   (🟡 More frequent logging)
```

### `scheduler_conf.warmup_steps`
```
Pretrained:           2500 (Long warmup for stability in large-scale training)
config_8k_telephone:  1000 (🟡 Shorter warmup)
stage1_encoder_adapt: 1000 (🟡 Shorter warmup)
```

---

## 5️⃣ DATASET CONFIGURATION - MEMORY OPTIMIZED

### `dataset_conf.max_token_length`
```
Pretrained:           3500
config_8k_telephone:  1024 (🟡 Much shorter - memory constraint)
stage1_encoder_adapt: 3500 (Uses pretrained default)
```
**Note**: Shorter sequences in main config to fit in GPU memory.

### `dataset_conf.audio_encoder_downsample_rate`
```
Pretrained:           6    (6:1 downsampling)
config_8k_telephone:  2    (🟡 2:1 downsampling)
stage1_encoder_adapt: 2    (🟡 2:1 downsampling)
```

### `dataset_conf.num_workers`
```
Pretrained:           4
config_8k_telephone:  4
stage1_encoder_adapt: 8    (🟡 More workers for 8kHz processing)
```

### `dataset_conf.retry`
```
Pretrained:           2000 (Many retries in original training)
config_8k_telephone:  20   (🟡 Few retries - production-ready data)
stage1_encoder_adapt: 2000 (Uses pretrained default)
```

---

## 6️⃣ OPTIMIZER - REGULARIZATION ADDED

### `optim_conf.weight_decay`
```
Pretrained:           0.0   (No regularization)
config_8k_telephone:  0.01  (🟡 Add L2 regularization)
stage1_encoder_adapt: 0.01  (🟡 Add L2 regularization)
```
**Why added**: Prevent overfitting during fine-tuning on smaller datasets.

### `optim_conf.betas` (New)
```
Pretrained:           [Not specified - uses Adam default]
config_8k_telephone:  [0.9, 0.999] (🟠 NEW - explicit specification)
stage1_encoder_adapt: [Uses code default]
```

---

## 7️⃣ SPECAUGMENTATION - ADDED FOR FINETUNING

### `specaug` (New parameter)
```
Pretrained:           [Not specified]
config_8k_telephone:  SpecAugLFR (🟠 NEW)
stage1_encoder_adapt: SpecAugLFR (🟠 NEW)
```

### `specaug_conf.freq_mask_width_range`
```
Pretrained:           [Not specified]
config_8k_telephone:  [0, 15]  (🟠 NEW - adapted for 8kHz)
stage1_encoder_adapt: [0, 10]  (🟠 NEW - smaller for narrow 8kHz spectrum)
```

### `specaug_conf.time_mask_width_range`
```
Pretrained:           [Not specified]
config_8k_telephone:  [0, 10]  (🟠 NEW)
stage1_encoder_adapt: [0, 50]  (🟠 NEW - 5x wider, simulate VoIP packet loss)
```

### `specaug_conf.num_time_mask`
```
Pretrained:           [Not specified]
config_8k_telephone:  1        (🟠 NEW)
stage1_encoder_adapt: 2        (🟠 NEW - more aggressive augmentation)
```

---

## 8️⃣ FROZEN COMPONENTS (UNCHANGED)

These remain frozen across all configurations:

| Component | Freeze Status | Trainable Parameters |
|-----------|---------------|----------------------|
| LLM (Qwen3-0.6B) | ✅ FROZEN | 0 |
| Audio Adaptor | ✅ FROZEN | 0 |
| ~~CTC Decoder~~ | Varies by stage | See section 2️⃣ |

---

## 9️⃣ PARAMETERS OMITTED (USE CODE DEFAULTS)

These are **NOT specified** in fine-tuning configs, so they use code defaults:

```
✓ audio_encoder_conf: attention_heads, linear_units, num_blocks, dropout_rate, etc.
✓ llm_conf: init_param_path, llm_dtype (use pretrained defaults)
✓ ctc_decoder_conf: ffn_dim, llm_dim, encoder_dim, n_layer (use pretrained)
✓ frontend_conf: cmvn_file (not used)
✓ train_conf: resume, save_init_model, effective_save_name_excludes
✓ dataset_conf: prompt_classes, ctc_tokenizer (uses pretrained settings)
```

---

## 🔟 PARAMETER SOURCES BY STAGE

### Stage 1: Encoder Adaptation (`stage1_encoder_adapt.yaml`)
**Trainable**: Only Audio Encoder
**Strategy**: Full encoder fine-tuning for 8kHz adaptation
**Key settings**:
- `audio_encoder_conf.freeze: false` ⚠️ **CRITICAL**
- `ctc_decoder_conf.freeze: true` (NOT trained in this stage)
- `optim_conf.lr: 1e-4` (Higher LR for encoder-only training)
- `frontend_conf.fs: 8000` (Native 8kHz, not upsampled)
- Heavy time augmentation (`time_mask_width_range: [0, 50]`)

### Main Config: CTC Decoder Fine-tuning (`config_8k_telephone.yaml`)
**Trainable**: Audio Encoder + CTC Decoder
**Strategy**: Full fine-tuning with upsampled 8kHz audio
**Key settings**:
- `audio_encoder_conf.freeze: false`
- `ctc_decoder_conf.freeze: false` (CTC IS trained)
- `optim_conf.lr: 5e-5` (Lower LR for multi-component training)
- `frontend_conf.fs: 16000` (Audio upsampled from 8kHz to 16kHz)
- Light time augmentation (`time_mask_width_range: [0, 10]`)

---

## ⚡ Training Impact Summary

| Aspect | Pretrained | Fine-tuning |
|--------|-----------|------------|
| Encoder trainable | ❌ Frozen | ✅ Active |
| CTC Decoder trainable | ✅ Active | ⚠️ Varies |
| Learning rate | 5e-6 | 50-200x higher |
| Epochs | 2 | 10-20 |
| Gradient accumulation | 1 | 1 |
| Weight decay | 0 | 0.01 |
| DeepSpeed | Enabled | Disabled |
| Batch size | 6000 tokens | 6000 tokens |
| Max sequence length | 3500 | 1024 (memory) |

---

## ✅ Pre-training Checklist

Before fine-tuning, verify:

- [ ] Pretrained weights loaded: `FunAudioLLM/Fun-ASR-Nano-2512/model.pt`
- [ ] Audio encoder freeze status: `false` for both configs ✓
- [ ] Frontend sample rate: Check stage-specific value (8000 or 16000)
- [ ] DeepSpeed disabled: `false` ✓
- [ ] Learning rate appropriate: Check for your dataset size
- [ ] SpecAug enabled: For data augmentation ✓
- [ ] CTC decoder status: `false` in config_8k_telephone, `true` in stage1

