# Configuration Alignment for Fun-ASR-Nano 8kHz Telephone Fine-tuning

## Overview
This document provides a detailed mapping of all configuration parameters between the **pretrained model** (`/gpfs01/nfs_share/finrc/liangguang/cache/modelscope/models/FunAudioLLM/Fun-ASR-Nano-2512/config.yaml`) and the **fine-tuning experiments** in this directory.

### Legend
- 🔵 **ALIGNED**: Same as pretrained model (no modification)
- 🟡 **MODIFIED**: Changed for fine-tuning (highlighted)
- 🟠 **NEW**: Not in pretrained config (added for fine-tuning)
- ⚪ **OMITTED**: Not set in fine-tuning (uses code default)

---

## 1. Network Architecture

### 1.1 Model Definition
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `model` | `FunASRNano` | 🔵 `FunASRNano` | 🔵 `FunASRNano` | Same model type |
| `model_conf.lsm_weight` | `0.1` | 🔵 `0.1` | 🔵 `0.1` | Label smoothing |
| `model_conf.length_normalized_loss` | `true` | 🔵 `true` | 🔵 `true` | Loss normalization |

---

## 2. Audio Encoder Configuration

### 2.1 Audio Encoder Type & Basic Config
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `audio_encoder` | `SenseVoiceEncoderSmall` | 🔵 Same | 🔵 Same | Same encoder type |
| `audio_encoder_conf.output_size` | `512` | 🔵 `512` | 🔵 `512` | Output dimension |
| `audio_encoder_conf.attention_heads` | `4` | ⚪ OMITTED | ⚪ OMITTED | Uses default from pretrained |
| `audio_encoder_conf.linear_units` | `2048` | ⚪ OMITTED | ⚪ OMITTED | Uses default from pretrained |
| `audio_encoder_conf.num_blocks` | `50` | ⚪ OMITTED | ⚪ OMITTED | Uses default from pretrained |
| `audio_encoder_conf.tp_blocks` | `20` | ⚪ OMITTED | ⚪ OMITTED | Uses default from pretrained |

### 2.2 Audio Encoder Fine-tuning Strategy
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `audio_encoder_conf.freeze` | 🔵 `true` | 🟡 **`false`** | 🟡 **`false`** | 🔴 **CRITICAL**: Encoder trainable in fine-tuning |
| `audio_encoder_conf.freeze_layer_num` | 🔵 `-1` | ⚪ OMITTED | ⚪ OMITTED | Not used by FunASRNano |
| `audio_encoder_conf.dropout_rate` | `0.1` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.positional_dropout_rate` | `0.1` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.attention_dropout_rate` | `0.1` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.input_layer` | `pe` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.pos_enc_class` | `SinusoidalPositionEncoder` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.normalize_before` | `true` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.kernel_size` | `11` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.sanm_shfit` | `0` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.selfattention_layer_type` | `sanm` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `audio_encoder_conf.feat_permute` | `true` | ⚪ OMITTED | ⚪ OMITTED | Uses default |

---

## 3. LLM Configuration

### 3.1 LLM Model & Setup
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `llm` | `Qwen3-0.6b` | 🔵 `Qwen3-0.6b` | 🔵 `Qwen3-0.6b` | Same LLM model |
| `llm_conf.hub` | `hf` | 🔵 `hf` | ⚪ OMITTED | HuggingFace hub (uses default) |
| `llm_conf.freeze` | `true` | 🔵 `true` | 🔵 `true` | LLM is frozen (not trainable) |
| `llm_conf.llm_dtype` | `bf16` | 🔵 `bf16` | ⚪ OMITTED | BFloat16 precision (uses default) |
| `llm_conf.init_param_path` | `Qwen3-0.6B` | 🔵 `Qwen3-0.6B` | ⚪ OMITTED | LLM weights path (uses default) |

### 3.2 LLM LoRA Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `llm_conf.use_lora` | `false` | ⚪ OMITTED | ⚪ OMITTED | No LoRA (uses default) |
| `llm_conf.lora_conf.*` | All defined | ⚪ OMITTED | ⚪ OMITTED | LoRA disabled |

---

## 4. Audio Adaptor Configuration

### 4.1 Adaptor Type & Architecture
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `audio_adaptor` | `Transformer` | 🔵 `Transformer` | 🔵 `Transformer` | Same adaptor type |
| `audio_adaptor_conf.downsample_rate` | `1` | 🔵 `1` | ⚪ OMITTED | No downsampling |
| `audio_adaptor_conf.use_low_frame_rate` | `true` | 🔵 `true` | ⚪ OMITTED | Uses default |
| `audio_adaptor_conf.ffn_dim` | `2048` | 🔵 `2048` | ⚪ OMITTED | FFN dimension |
| `audio_adaptor_conf.llm_dim` | `1024` | 🔵 `1024` | ⚪ OMITTED | LLM dimension |
| `audio_adaptor_conf.encoder_dim` | `512` | 🔵 `512` | ⚪ OMITTED | Encoder dimension |
| `audio_adaptor_conf.n_layer` | `2` | 🔵 `2` | ⚪ OMITTED | Number of layers |

### 4.2 Adaptor Fine-tuning Strategy
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `audio_adaptor_conf.freeze` | `true` | 🔵 `true` | 🔵 `true` | Adaptor is frozen (not trainable) |

---

## 5. CTC Decoder Configuration

### 5.1 Decoder Type & Architecture
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `ctc_decoder` | `Transformer` | 🔵 `Transformer` | 🔵 `Transformer` | Same decoder type |
| `ctc_decoder_conf.downsample_rate` | `1` | ⚪ OMITTED | ⚪ OMITTED | No downsampling |
| `ctc_decoder_conf.ffn_dim` | `2048` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_decoder_conf.llm_dim` | `512` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_decoder_conf.encoder_dim` | `512` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_decoder_conf.n_layer` | `5` | ⚪ OMITTED | ⚪ OMITTED | Uses default |

### 5.2 CTC Decoder Fine-tuning Strategy
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `ctc_decoder_conf.freeze` | `false` | 🔵 `false` | 🔵 `true` | 🟡 **DIFFERENT STAGES**: config_8k_telephone trains CTC; stage1 freezes it |
| `detach_ctc_decoder` | `true` | ⚪ OMITTED | ⚪ OMITTED | Uses default |

### 5.3 CTC Weight
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `ctc_weight` | `1.0` | 🔵 `1.0` | ⚪ OMITTED | CTC loss weight |
| `ctc_conf.dropout_rate` | `0.0` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_conf.ctc_type` | `builtin` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_conf.reduce` | `true` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `ctc_conf.ignore_nan_grad` | `true` | ⚪ OMITTED | ⚪ OMITTED | Uses default |

---

## 6. Frontend Configuration

### 6.1 Frontend Type & Sampling Rate
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `frontend` | `WavFrontend` | 🔵 `WavFrontend` | 🔵 `WavFrontend` | Same frontend type |
| `frontend_conf.fs` | `16000` | 🟡 **`16000`** | 🟡 **`8000`** | 🔴 **CRITICAL**: Sample rate differs by stage |
| `frontend_conf.window` | `hamming` | 🔵 `hamming` | 🔵 `hamming` | Window type |
| `frontend_conf.n_mels` | `80` | 🔵 `80` | 🔵 `80` | Mel features |
| `frontend_conf.frame_length` | `25` | 🔵 `25` | 🔵 `25` | Frame length (ms) |
| `frontend_conf.frame_shift` | `10` | 🔵 `10` | 🔵 `10` | Frame shift (ms) |
| `frontend_conf.lfr_m` | `7` | 🔵 `7` | 🔵 `7` | LFR mode |
| `frontend_conf.lfr_n` | `6` | 🔵 `6` | 🔵 `6` | LFR factor |
| `frontend_conf.cmvn_file` | `null` | ⚪ OMITTED | ⚪ OMITTED | No CMVN normalization |

### 6.2 SpecAugmentation
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `specaug` | ⚪ NOT SET | 🟡 **`SpecAugLFR`** | 🟡 **`SpecAugLFR`** | 🟠 **NEW**: SpecAug added for fine-tuning |
| `specaug_conf.apply_time_warp` | ⚪ N/A | 🟡 `false` | 🟡 `false` | No time warping |
| `specaug_conf.time_warp_window` | ⚪ N/A | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `specaug_conf.apply_freq_mask` | ⚪ N/A | 🟡 `true` | 🟡 `true` | Frequency masking enabled |
| `specaug_conf.freq_mask_width_range` | ⚪ N/A | 🟡 `[0, 15]` | 🟡 `[0, 10]` | 🟡 **MODIFIED**: Smaller range for 8kHz |
| `specaug_conf.lfr_rate` | ⚪ N/A | 🟡 `6` | 🟡 `6` | LFR rate |
| `specaug_conf.num_freq_mask` | ⚪ N/A | 🟡 `1` | 🟡 `1` | Number of freq masks |
| `specaug_conf.apply_time_mask` | ⚪ N/A | 🟡 `true` | 🟡 `true` | Time masking enabled |
| `specaug_conf.time_mask_width_range` | ⚪ N/A | 🟡 `[0, 10]` | 🟡 `[0, 50]` | 🟡 **MODIFIED**: Different ranges for different stages |
| `specaug_conf.num_time_mask` | ⚪ N/A | 🟡 `1` | 🟡 `2` | 🟡 **MODIFIED**: More masking in stage1 |

---

## 7. Training Configuration

### 7.1 Basic Training Parameters
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `train_conf.accum_grad` | `1` | 🔵 `1` | 🔵 `1` | Gradient accumulation steps |
| `train_conf.grad_clip` | `5` | 🔵 `5` | 🔵 `5` | Gradient clipping value |
| `train_conf.max_epoch` | `2` | 🟡 **`20`** | 🟡 **`10`** | 🔴 **MODIFIED**: More epochs for fine-tuning |
| `train_conf.keep_nbest_models` | `200` | 🟡 **`10`** | 🟡 **`5`** | 🟡 **MODIFIED**: Fewer models kept |
| `train_conf.log_interval` | `100` | 🟡 **`10`** | 🟡 **`10`** | 🟡 **MODIFIED**: More frequent logging |

### 7.2 Checkpoint & Validation Strategy
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `train_conf.validate_interval` | `2000` | ⚪ OMITTED | 🟡 `1000` | 🔴 **stage1**: More frequent validation |
| `train_conf.save_checkpoint_interval` | `2000` | ⚪ OMITTED | 🟡 `1000` | 🔴 **stage1**: More frequent checkpoints |
| `train_conf.avg_nbest_model` | `100` | ⚪ OMITTED | 🟡 `3` | 🟡 **stage1**: Model averaging |
| `train_conf.use_bf16` | `false` | ⚪ OMITTED | 🟡 `false` | No mixed precision |
| `train_conf.use_deepspeed` | `true` | 🟡 **`false`** | 🟡 **`false`** | 🟡 **MODIFIED**: DeepSpeed disabled for fine-tuning |
| `train_conf.deepspeed_config` | `null` | 🔵 `null` | ⚪ OMITTED | No DeepSpeed config |
| `train_conf.save_init_model` | `false` | ⚪ OMITTED | ⚪ OMITTED | Uses default |
| `train_conf.effective_save_name_excludes` | `[llm.]` | ⚪ OMITTED | ⚪ OMITTED | Exclude LLM from checkpoints |
| `train_conf.resume` | `true` | ⚪ OMITTED | ⚪ OMITTED | Resume training |

---

## 8. Optimizer Configuration

### 8.1 Optimizer Type & Hyperparameters
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `optim` | `adamw` | 🔵 `adamw` | 🔵 `adamw` | Same optimizer |
| `optim_conf.lr` | `5.0e-06` | 🟡 **`5e-5`** | 🟡 **`1e-4`** | 🔴 **MODIFIED**: Higher LR for fine-tuning |
| `optim_conf.weight_decay` | `0.0` | 🟡 **`0.01`** | 🟡 **`0.01`** | 🔴 **MODIFIED**: Added weight decay |
| `optim_conf.betas` | ⚪ N/A | 🟡 `[0.9, 0.999]` | ⚪ OMITTED | 🟠 **NEW**: Beta parameters specified in config_8k_telephone |

---

## 9. Learning Rate Scheduler

### 9.1 Scheduler Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `scheduler` | `warmuplr` | 🔵 `warmuplr` | 🔵 `warmuplr` | Same scheduler type |
| `scheduler_conf.warmup_steps` | `2500` | 🟡 **`1000`** | 🟡 **`1000`** | 🟡 **MODIFIED**: Shorter warmup |

---

## 10. Dataset Configuration

### 10.1 Dataset Type & Batch Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `dataset` | `FunASR` | 🔵 `FunASR` | 🔵 `FunASR` | Same dataset type |
| `dataset_conf.index_ds` | `FunASR` | 🔵 `FunASR` | 🔵 `FunASR` | Index dataset type |
| `dataset_conf.batch_sampler` | `BatchSampler` | 🔵 `BatchSampler` | 🔵 `BatchSampler` | Batch sampling strategy |
| `dataset_conf.batch_type` | `token` | 🔵 `token` | 🔵 `token` | Token-based batching |
| `dataset_conf.batch_size` | `6000` | 🔵 `6000` | 🔵 `6000` | Batch size (tokens) |
| `dataset_conf.max_token_length` | `3500` | 🟡 **`1024`** | 🔵 `3500` | 🟡 **MODIFIED**: Shorter max length in config_8k_telephone |
| `dataset_conf.shuffle` | `true` | 🔵 `true` | 🔵 `true` | Shuffle data |
| `dataset_conf.sort_size` | `1024` | 🔵 `1024` | 🔵 `1024` | Sort buffer size |
| `dataset_conf.batch_size_scale_ratio_max` | `2` | 🔵 `2` | ⚪ OMITTED | Uses default |

### 10.2 Data Loading & Processing
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `dataset_conf.num_workers` | `4` | 🔵 `4` | 🟡 **`8`** | 🟡 **MODIFIED**: More workers in stage1 |
| `dataset_conf.audio_adaptor_downsample_rate` | `${audio_adaptor_conf.downsample_rate}` | 🔵 Same | 🔵 Same | References adaptor config |
| `dataset_conf.audio_encoder_downsample_rate` | `6` | 🟡 **`2`** | 🔵 `2` | 🟡 **MODIFIED**: Different downsampling rate |
| `dataset_conf.data_split_num` | `256` | 🟡 **`512`** | 🔵 `256` | 🟡 **MODIFIED**: More splits in config_8k_telephone |
| `dataset_conf.batch_size_sample_max` | `10` | 🟡 **`15`** | 🔵 `10` | 🟡 **MODIFIED**: Larger sample batch in config_8k_telephone |
| `dataset_conf.retry` | `2000` | 🟡 **`20`** | 🔵 `2000` | 🟡 **MODIFIED**: Fewer retries in config_8k_telephone |

### 10.3 Sequence Length Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `dataset_conf.batch_size_token_max` | `6000` | ⚪ OMITTED | 🔵 `6000` | Max tokens per batch |
| `dataset_conf.max_source_length` | `12000` | ⚪ OMITTED | 🔵 `12000` | Max source length |
| `dataset_conf.max_target_length` | `2048` | ⚪ OMITTED | 🔵 `2048` | Max target length |
| `dataset_conf.min_source_length` | `10` | ⚪ OMITTED | 🔵 `10` | Min source length |
| `dataset_conf.batch_size_scale_threshold` | `3000` | ⚪ OMITTED | 🔵 `3000` | Batch scaling threshold |

### 10.4 Prompt & Hotword Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `dataset_conf.prompt_classes` | `MultiContextPrompt` | ⚪ OMITTED | 🔵 `MultiContextPrompt` | Multi-context prompting |
| `dataset_conf.prompt_conf.max_neg_hotwords_num` | `0` | ⚪ OMITTED | 🔵 `0` | Max negative hotwords |
| `dataset_conf.prompt_conf.min_neg_hotwords_num` | `0` | ⚪ OMITTED | 🔵 `0` | Min negative hotwords |
| `dataset_conf.prompt_conf.use_hist` | `false` | ⚪ OMITTED | 🔵 `false` | Use history |
| `dataset_conf.prompt_conf.use_one_pass_result` | `true` | ⚪ OMITTED | 🔵 `true` | Use one-pass result |
| `dataset_conf.prompt_conf.use_hotwords` | `true` | ⚪ OMITTED | 🔵 `true` | Use hotwords |
| `dataset_conf.prompt_conf.use_asr_hotwords` | `true` | ⚪ OMITTED | 🔵 `true` | Use ASR hotwords |
| `dataset_conf.prompt_conf.chinese_hotwords_list` | `null` | ⚪ OMITTED | 🔵 `null` | Chinese hotwords |
| `dataset_conf.prompt_conf.english_hotwords_list` | `null` | ⚪ OMITTED | 🔵 `null` | English hotwords |

### 10.5 CTC Tokenizer Configuration
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `dataset_conf.ctc_tokenizer` | `SenseVoiceTokenizer` | ⚪ OMITTED | 🔵 `SenseVoiceTokenizer` | CTC tokenizer type |
| `dataset_conf.ctc_target_normalize` | `true` | ⚪ OMITTED | 🔵 `true` | Normalize CTC targets |
| `dataset_conf.ctc_tokenizer_conf.vocab_path` | `null` | ⚪ OMITTED | 🔵 `null` | Vocabulary path |
| `dataset_conf.ctc_tokenizer_conf.is_multilingual` | `true` | ⚪ OMITTED | 🔵 `true` | Multilingual support |
| `dataset_conf.ctc_tokenizer_conf.num_languages` | `8749` | ⚪ OMITTED | 🔵 `8749` | Number of languages |
| `dataset_conf.use_dynamic_output_ratio` | `0.0` | ⚪ OMITTED | 🔵 `0.0` | Dynamic output ratio |

---

## 11. Tokenizer Configuration

### 11.1 Tokenizer Type & Setup
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `tokenizer` | `HuggingfaceTokenizer` | 🔵 `HuggingfaceTokenizer` | 🔵 `HuggingfaceTokenizer` | Same tokenizer type |
| `tokenizer_conf.init_param_path` | `${llm_conf.init_param_path}` | 🟡 `Qwen3-0.6B` | 🟡 `Qwen3-0.6B` | 🟡 **MODIFIED**: Explicit reference instead of variable |
| `tokenizer_conf.unk_symbol` | ⚪ N/A | 🟠 **`<unk>`** | 🟠 **`<unk>`** | 🟠 **NEW**: Unknown token symbol |

---

## 12. Other Configuration

### 12.1 Miscellaneous Settings
| Parameter | Pretrained | config_8k_telephone | stage1_encoder_adapt | Notes |
|-----------|-----------|------------------|---------------------|-------|
| `enable_tf32` | `true` | ⚪ OMITTED | ⚪ OMITTED | TF32 acceleration |
| `debug` | `false` | ⚪ OMITTED | ⚪ OMITTED | Debug mode |
| `train_data_set_list` | `null` | ⚪ OMITTED | ⚪ OMITTED | Training dataset list |
| `valid_data_set_list` | `null` | ⚪ OMITTED | ⚪ OMITTED | Validation dataset list |
| `init_param` | `null` | ⚪ OMITTED | ⚪ OMITTED | Initial parameters path |
| `output_dir` | `null` | ⚪ OMITTED | ⚪ OMITTED | Output directory |

---

## Summary of Key Changes for Fine-tuning

### 🔴 Critical Modifications
1. **Audio Encoder Freeze** (`audio_encoder_conf.freeze`): `true` → `false`
   - Encoder becomes trainable during fine-tuning
   
2. **Frontend Sample Rate** (`frontend_conf.fs`): 
   - Pretrained: `16000`
   - config_8k_telephone: `16000`
   - stage1_encoder_adapt: `8000` ← Different by stage

3. **CTC Decoder Freeze** (`ctc_decoder_conf.freeze`):
   - config_8k_telephone: `false` (trainable)
   - stage1_encoder_adapt: `true` (frozen)

4. **Learning Rate** (`optim_conf.lr`):
   - Pretrained: `5e-6` (very small for inference)
   - config_8k_telephone: `5e-5` (10× higher)
   - stage1_encoder_adapt: `1e-4` (20× higher)

5. **Training Epochs** (`train_conf.max_epoch`):
   - Pretrained: `2`
   - Fine-tuning: `10-20` epochs

6. **DeepSpeed** (`train_conf.use_deepspeed`):
   - Pretrained: `true`
   - Fine-tuning: `false` (disabled for smaller experiments)

### 🟡 Dataset/Training Adjustments
- **Batch size & sequence length**: Adjusted for memory constraints
- **SpecAugmentation**: Added for data augmentation (not in pretrained)
- **Warmup steps**: Reduced from `2500` to `1000`
- **Weight decay**: Enabled for regularization
- **Data loading workers**: Adjusted by stage (4-8)

### ⚪ Omitted Parameters (Use Code Defaults)
Many parameters not specified in fine-tuning configs will use defaults from:
- Base configuration file
- Model class defaults
- Code implementation

This includes encoder-specific parameters (dropout rates, attention heads, etc.) that remain unchanged from pretrained model.

---

## How to Use This Document

1. **Verify alignment before training**: Compare your config with the pretrained model
2. **Understand modifications**: Each 🟡 marked change has been intentionally modified for fine-tuning
3. **Check omitted parameters**: If a parameter is not in your config (⚪), verify code defaults are acceptable
4. **Stage-specific differences**: Note differences between stage1_encoder_adapt and config_8k_telephone
5. **Document your customizations**: If you modify any parameters, update this document

