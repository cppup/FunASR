# 配置对齐映射 - 代码即文档

## 概述

所有配置文件已根据预训练模型进行**代码化对齐**：
- ✅ 所有预训练模型中存在的配置项都**显式写入**
- 🔵 **[ALIGNED]** 标记：与预训练模型相同的参数
- 🟡 **[MODIFIED]** 标记：为微调修改的参数  
- 🟠 **[NEW]** 标记：预训练中不存在的新增参数
- ❌ **拒绝隐式默认值**：所有参数都明确指定

## 预训练模型配置源

```
/gpfs01/nfs_share/finrc/liangguang/cache/modelscope/models/
    FunAudioLLM/Fun-ASR-Nano-2512/config.yaml
```

## 配置文件修改统计

### config_8k_telephone.yaml (主配置)
```
总行数:        274 行
[ALIGNED]:     126 项 (基本保持预训练配置)
[MODIFIED]:    20 项 (为8kHz微调修改)
[NEW]:         14 项 (新增微调参数)
```

**关键修改**:
1. `audio_encoder_conf.freeze`: true → false (启用encoder训练)
2. `optim_conf.lr`: 5e-6 → 5e-5 (学习率提升10倍)
3. `train_conf.use_deepspeed`: true → false (单GPU模式)
4. `specaug`: NEW (SpecAugLFR数据增强)
5. `optim_conf.weight_decay`: 0.0 → 0.01 (正则化)

### stage1_encoder_adapt.yaml (Stage1配置)
```
总行数:        292 行
[ALIGNED]:     140 项 (基本保持预训练配置)
[MODIFIED]:    17 项 (为encoder适应修改)
[NEW]:         4 项 (新增参数)
```

**关键修改**:
1. `frontend_conf.fs`: 16000 → 8000 (原生8kHz处理)
2. `ctc_decoder_conf.freeze`: false → true (冻结CTC)
3. `optim_conf.lr`: 5e-6 → 1e-4 (学习率提升20倍)
4. `specaug_conf.time_mask_width_range`: [0,10] → [0,50] (VoIP丢帧模拟)

## 详细对齐映射

### 1. MODEL ARCHITECTURE (模型架构)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| model | FunASRNano | FunASRNano | FunASRNano | ✅ ALIGNED |
| model_conf.lsm_weight | 0.1 | 0.1 | 0.1 | ✅ ALIGNED |
| model_conf.length_normalized_loss | true | true | true | ✅ ALIGNED |

### 2. AUDIO ENCODER (音频编码器)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| audio_encoder | SenseVoiceEncoderSmall | SenseVoiceEncoderSmall | SenseVoiceEncoderSmall | ✅ ALIGNED |
| output_size | 512 | 512 | 512 | ✅ ALIGNED |
| attention_heads | 4 | 4 | 4 | ✅ ALIGNED |
| linear_units | 2048 | 2048 | 2048 | ✅ ALIGNED |
| num_blocks | 50 | 50 | 50 | ✅ ALIGNED |
| tp_blocks | 20 | 20 | 20 | ✅ ALIGNED |
| dropout_rate | 0.1 | 0.1 | 0.1 | ✅ ALIGNED |
| positional_dropout_rate | 0.1 | 0.1 | 0.1 | ✅ ALIGNED |
| attention_dropout_rate | 0.1 | 0.1 | 0.1 | ✅ ALIGNED |
| input_layer | pe | pe | pe | ✅ ALIGNED |
| pos_enc_class | SinusoidalPositionEncoder | SinusoidalPositionEncoder | SinusoidalPositionEncoder | ✅ ALIGNED |
| normalize_before | true | true | true | ✅ ALIGNED |
| kernel_size | 11 | 11 | 11 | ✅ ALIGNED |
| sanm_shfit | 0 | 0 | 0 | ✅ ALIGNED |
| selfattention_layer_type | sanm | sanm | sanm | ✅ ALIGNED |
| feat_permute | true | true | true | ✅ ALIGNED |
| **freeze** | **true** | **false** | **false** | 🟡 MODIFIED |
| freeze_layer_num | -1 | -1 | -1 | ✅ ALIGNED |

### 3. LLM (语言模型)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| llm | Qwen3-0.6b | Qwen3-0.6b | Qwen3-0.6b | ✅ ALIGNED |
| hub | hf | hf | hf | ✅ ALIGNED |
| freeze | true | true | true | ✅ ALIGNED |
| llm_dtype | bf16 | bf16 | bf16 | ✅ ALIGNED |
| init_param_path | Qwen3-0.6B | Qwen3-0.6B | Qwen3-0.6B | ✅ ALIGNED |
| use_lora | false | false | false | ✅ ALIGNED |
| (lora_conf.*) | (all defined) | (all defined) | (all defined) | ✅ ALIGNED |

### 4. AUDIO ADAPTOR (音频适配器)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| audio_adaptor | Transformer | Transformer | Transformer | ✅ ALIGNED |
| downsample_rate | 1 | 1 | 1 | ✅ ALIGNED |
| use_low_frame_rate | true | true | true | ✅ ALIGNED |
| ffn_dim | 2048 | 2048 | 2048 | ✅ ALIGNED |
| llm_dim | 1024 | 1024 | 1024 | ✅ ALIGNED |
| encoder_dim | 512 | 512 | 512 | ✅ ALIGNED |
| n_layer | 2 | 2 | 2 | ✅ ALIGNED |
| freeze | true | true | true | ✅ ALIGNED |

### 5. CTC DECODER (CTC解码器)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| ctc_decoder | Transformer | Transformer | Transformer | ✅ ALIGNED |
| detach_ctc_decoder | true | true | true | ✅ ALIGNED |
| downsample_rate | 1 | 1 | 1 | ✅ ALIGNED |
| ffn_dim | 2048 | 2048 | 2048 | ✅ ALIGNED |
| llm_dim | 512 | 512 | 512 | ✅ ALIGNED |
| encoder_dim | 512 | 512 | 512 | ✅ ALIGNED |
| n_layer | 5 | 5 | 5 | ✅ ALIGNED |
| **freeze** | **false** | **false** | **true** | 🟡 MODIFIED (stage1) |
| ctc_weight | 1.0 | 1.0 | 1.0 | ✅ ALIGNED |
| (ctc_conf.*) | (all defined) | (all defined) | (all defined) | ✅ ALIGNED |

### 6. FRONTEND (前端处理)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| frontend | WavFrontend | WavFrontend | WavFrontend | ✅ ALIGNED |
| fs | 16000 | 16000 | **8000** | 🟡 MODIFIED (stage1) |
| window | hamming | hamming | hamming | ✅ ALIGNED |
| n_mels | 80 | 80 | 80 | ✅ ALIGNED |
| frame_length | 25 | 25 | 25 | ✅ ALIGNED |
| frame_shift | 10 | 10 | 10 | ✅ ALIGNED |
| lfr_m | 7 | 7 | 7 | ✅ ALIGNED |
| lfr_n | 6 | 6 | 6 | ✅ ALIGNED |
| cmvn_file | null | null | null | ✅ ALIGNED |

### 7. SPECAUGMENT (NEW - 数据增强)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| specaug | ❌ 无 | SpecAugLFR | SpecAugLFR | 🟠 NEW |
| apply_time_warp | ❌ 无 | false | false | 🟠 NEW |
| apply_freq_mask | ❌ 无 | true | true | 🟠 NEW |
| freq_mask_width_range | ❌ 无 | [0,15] | [0,10] | 🟠 NEW |
| apply_time_mask | ❌ 无 | true | true | 🟠 NEW |
| time_mask_width_range | ❌ 无 | [0,10] | **[0,50]** | 🟠 NEW |
| num_time_mask | ❌ 无 | 1 | **2** | 🟠 NEW |

### 8. TRAINING CONFIGURATION (训练配置)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| accum_grad | 1 | 1 | 1 | ✅ ALIGNED |
| grad_clip | 5 | 5 | 5 | ✅ ALIGNED |
| max_epoch | **2** | **20** | **10** | 🟡 MODIFIED |
| keep_nbest_models | **200** | **10** | **5** | 🟡 MODIFIED |
| log_interval | **100** | **10** | **10** | 🟡 MODIFIED |
| validate_interval | 2000 | 2000 | **1000** | 🟡 MODIFIED (stage1) |
| save_checkpoint_interval | 2000 | 2000 | **1000** | 🟡 MODIFIED (stage1) |
| avg_nbest_model | 100 | 100 | **3** | 🟡 MODIFIED (stage1) |
| use_bf16 | false | false | false | ✅ ALIGNED |
| **use_deepspeed** | **true** | **false** | **false** | 🟡 MODIFIED |
| deepspeed_config | null | null | null | ✅ ALIGNED |
| save_init_model | false | false | false | ✅ ALIGNED |

### 9. OPTIMIZER CONFIGURATION (优化器)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| optim | adamw | adamw | adamw | ✅ ALIGNED |
| **lr** | **5e-6** | **5e-5** | **1e-4** | 🟡 MODIFIED |
| **weight_decay** | **0.0** | **0.01** | **0.01** | 🟡 MODIFIED |

### 10. SCHEDULER CONFIGURATION (学习率调度)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| scheduler | warmuplr | warmuplr | warmuplr | ✅ ALIGNED |
| **warmup_steps** | **2500** | **1000** | **1000** | 🟡 MODIFIED |

### 11. DATASET CONFIGURATION (数据集配置)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| dataset | FunASR | FunASR | FunASR | ✅ ALIGNED |
| index_ds | FunASR | FunASR | FunASR | ✅ ALIGNED |
| batch_sampler | BatchSampler | BatchSampler | BatchSampler | ✅ ALIGNED |
| batch_type | token | token | token | ✅ ALIGNED |
| batch_size | 6000 | 6000 | 6000 | ✅ ALIGNED |
| **max_token_length** | **3500** | **1024** | **3500** | 🟡 MODIFIED (config) |
| shuffle | true | true | true | ✅ ALIGNED |
| sort_size | 1024 | 1024 | 1024 | ✅ ALIGNED |
| batch_size_scale_ratio_max | 2 | 2 | 2 | ✅ ALIGNED |
| num_workers | 4 | 4 | **8** | 🟡 MODIFIED (stage1) |
| **audio_encoder_downsample_rate** | **6** | **2** | **2** | 🟡 MODIFIED |
| **data_split_num** | **256** | **512** | **256** | 🟡 MODIFIED (config) |
| **batch_size_sample_max** | **10** | **15** | **10** | 🟡 MODIFIED (config) |
| **retry** | **2000** | **20** | **2000** | 🟡 MODIFIED (config) |
| batch_size_token_max | 6000 | 6000 | 6000 | ✅ ALIGNED |
| max_source_length | 12000 | 12000 | 12000 | ✅ ALIGNED |
| max_target_length | 2048 | 2048 | 2048 | ✅ ALIGNED |
| prompt_classes | MultiContextPrompt | MultiContextPrompt | MultiContextPrompt | ✅ ALIGNED |
| (prompt_conf.*) | (all defined) | (all defined) | (all defined) | ✅ ALIGNED |
| ctc_tokenizer | SenseVoiceTokenizer | SenseVoiceTokenizer | SenseVoiceTokenizer | ✅ ALIGNED |
| ctc_target_normalize | true | true | true | ✅ ALIGNED |
| (ctc_tokenizer_conf.*) | (all defined) | (all defined) | (all defined) | ✅ ALIGNED |
| min_source_length | 10 | 10 | 10 | ✅ ALIGNED |
| batch_size_scale_threshold | 3000 | 3000 | 3000 | ✅ ALIGNED |
| use_dynamic_output_ratio | 0.0 | 0.0 | 0.0 | ✅ ALIGNED |

### 12. TOKENIZER CONFIGURATION (分词器)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| tokenizer | HuggingfaceTokenizer | HuggingfaceTokenizer | HuggingfaceTokenizer | ✅ ALIGNED |
| init_param_path | ${llm_conf.init_param_path} | ${llm_conf.init_param_path} | ${llm_conf.init_param_path} | ✅ ALIGNED |
| **unk_symbol** | ❌ 无 | **<unk>** | **<unk>** | 🟠 NEW |

### 13. GLOBAL SETTINGS (全局设置)

| 参数 | 预训练 | config_8k_telephone | stage1 | 状态 |
|------|------|-----------------|--------|------|
| enable_tf32 | true | true | true | ✅ ALIGNED |
| debug | false | false | false | ✅ ALIGNED |
| train_data_set_list | null | null | null | ✅ ALIGNED |
| valid_data_set_list | null | null | null | ✅ ALIGNED |
| init_param | null | null | null | ✅ ALIGNED |
| output_dir | null | null | null | ✅ ALIGNED |

## 修改统计总结

### config_8k_telephone.yaml

```
✅ ALIGNED (126项):
   - 完整的模型架构 (model, audio_encoder基础配置)
   - LLM (Qwen3-0.6B) 及其配置
   - Audio Adaptor 配置
   - CTC Decoder基础配置
   - Frontend基础配置
   - Dataset基础配置
   - Global设置

🟡 MODIFIED (20项):
   1. audio_encoder_conf.freeze: true → false
   2. train_conf.max_epoch: 2 → 20
   3. train_conf.keep_nbest_models: 200 → 10
   4. train_conf.log_interval: 100 → 10
   5. train_conf.use_deepspeed: true → false
   6. optim_conf.lr: 5e-6 → 5e-5
   7. optim_conf.weight_decay: 0.0 → 0.01
   8. scheduler_conf.warmup_steps: 2500 → 1000
   9. dataset_conf.max_token_length: 3500 → 1024
   10. dataset_conf.audio_encoder_downsample_rate: 6 → 2
   11. dataset_conf.data_split_num: 256 → 512
   12. dataset_conf.batch_size_sample_max: 10 → 15
   13. dataset_conf.retry: 2000 → 20
   + specaug_conf.freq_mask_width_range
   + specaug_conf.time_mask_width_range
   + specaug_conf.num_time_mask
   + 等等

🟠 NEW (14项):
   1. specaug: SpecAugLFR
   2. specaug_conf.apply_time_warp
   3. specaug_conf.time_warp_window
   4. specaug_conf.time_warp_mode
   5. specaug_conf.apply_freq_mask
   6. specaug_conf.freq_mask_width_range
   7. specaug_conf.lfr_rate
   8. specaug_conf.num_freq_mask
   9. specaug_conf.apply_time_mask
   10. specaug_conf.time_mask_width_range
   11. specaug_conf.num_time_mask
   12. optim_conf.betas
   13. tokenizer_conf.unk_symbol
   14. (其他新增)
```

### stage1_encoder_adapt.yaml

```
✅ ALIGNED (140项):
   - 同config_8k_telephone，但不同的冻结策略
   - 完整的预训练参数传承

🟡 MODIFIED (17项):
   1. audio_encoder_conf.freeze: true → false (训练encoder)
   2. ctc_decoder_conf.freeze: false → true (Stage1冻结CTC)
   3. frontend_conf.fs: 16000 → 8000 (原生8kHz)
   4. specaug_conf.time_mask_width_range: [0,10] → [0,50]
   5. specaug_conf.num_time_mask: 1 → 2
   6. train_conf.max_epoch: 2 → 10
   7. train_conf.keep_nbest_models: 200 → 5
   8. train_conf.log_interval: 100 → 10
   9. train_conf.validate_interval: 2000 → 1000
   10. train_conf.save_checkpoint_interval: 2000 → 1000
   11. train_conf.avg_nbest_model: 100 → 3
   12. optim_conf.lr: 5e-6 → 1e-4
   13. optim_conf.weight_decay: 0.0 → 0.01
   14. dataset_conf.num_workers: 4 → 8
   + 其他

🟠 NEW (4项):
   1. specaug: SpecAugLFR
   2. specaug_conf.* (所有SpecAug参数)
   3. tokenizer_conf.unk_symbol
```

## 代码即文档说明

每个配置文件中的每一行都包含以下信息之一：

### 示例：ALIGNED参数
```yaml
freeze: true  # [ALIGNED]
```
含义：该参数与预训练模型相同，未作修改

### 示例：MODIFIED参数
```yaml
freeze: false  # [MODIFIED] Pretrained=true, changed to enable encoder fine-tuning for 8kHz
```
含义：该参数已从预训练模型修改，说明了原值和修改原因

### 示例：NEW参数
```yaml
specaug: SpecAugLFR  # [NEW] Not in pretrained model, added for fine-tuning data augmentation
```
含义：该参数不存在于预训练模型，是为微调新增的

## 使用方式

### 1. 快速查看配置
```bash
# 查看所有ALIGNED参数
grep -n '\[ALIGNED\]' conf/config_8k_telephone.yaml

# 查看所有MODIFIED参数
grep -n '\[MODIFIED\]' conf/config_8k_telephone.yaml

# 查看所有NEW参数
grep -n '\[NEW\]' conf/config_8k_telephone.yaml
```

### 2. 对比两个配置
```bash
# 对比config和stage1的差异
diff -u conf/config_8k_telephone.yaml conf/stage1_encoder_adapt.yaml | grep -A1 -B1 MODIFIED

# 查看stage1特有的修改
diff conf/config_8k_telephone.yaml conf/stage1_encoder_adapt.yaml | grep '<' | grep MODIFIED
```

### 3. 理解特定参数
直接打开配置文件，搜索参数名，查看其标记和注释即可了解：
- 与预训练的关系
- 修改原因
- 修改值

## 验证清单

在使用配置前，验证以下关键项：

```
✓ [ALIGNED] 项确实与预训练模型一致
✓ [MODIFIED] 项的修改值正确
✓ [NEW] 项没有遗漏
✓ 没有隐式默认值 - 所有参数都显式指定
✓ 冻结策略正确：
  - LLM始终frozen=true
  - Adaptor始终frozen=true
  - Encoder在config和stage1都是frozen=false
  - CTC在config是frozen=false, stage1是frozen=true
```

## 总结

通过代码化对齐，这两个配置文件现在可以作为**动态文档**：
- ✅ 每个参数都有明确的来源说明
- ✅ 修改的原因都在注释中解释
- ✅ 没有隐式默认值 - 代码即文档
- ✅ 易于维护：修改时可以直接查看改动点
- ✅ 易于学习：新用户可以通过标记快速理解配置

