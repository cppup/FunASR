# Configuration Comparison Table

## 快速查看 - 预训练 vs 微调配置差异

### 图例
- 🟢 ALIGNED (同预训练)
- 🟡 MODIFIED (已修改)
- 🔴 CRITICAL (关键修改)
- 🟠 NEW (新增)
- ⚪ OMITTED (使用代码默认)

---

## 核心配置对比表

| 模块 | 参数 | 预训练模型 | config_8k_telephone | stage1_encoder_adapt | 说明 |
|------|------|----------|-------------------|-------------------|------|
| **Model** | model | FunASRNano | 🟢 FunASRNano | 🟢 FunASRNano | 保持一致 |
| | lsm_weight | 0.1 | 🟢 0.1 | 🟢 0.1 | 标签平滑 |
| **Encoder** | freeze | **true** | 🔴 **false** | 🔴 **false** | ⚠️ 编码器变为可训练 |
| | output_size | 512 | 🟢 512 | 🟢 512 | 输出维度 |
| **LLM** | model | Qwen3-0.6b | 🟢 Qwen3-0.6b | 🟢 Qwen3-0.6b | 保持冻结 |
| | freeze | true | 🟢 true | 🟢 true | 完全冻结 |
| **Adaptor** | freeze | true | 🟢 true | 🟢 true | 保持冻结 |
| **CTC Decoder** | freeze | false | 🟢 false | 🟡 **true** | 🔴 Stage1 冻结CTC |
| **Frontend** | fs | 16000 | 🟢 16000 | 🟡 **8000** | 🔴 Stage1 使用原生8kHz |
| | n_mels | 80 | 🟢 80 | 🟢 80 | Mel频谱 |
| **SpecAug** | type | ⚪ 无 | 🟠 SpecAugLFR | 🟠 SpecAugLFR | 新增数据增强 |
| | freq_mask_range | ⚪ 无 | 🟡 [0,15] | 🟡 [0,10] | 频率掩码范围 |
| | time_mask_range | ⚪ 无 | 🟡 [0,10] | 🟡 [0,50] | 时间掩码范围(VoIP模拟) |
| | num_time_mask | ⚪ 无 | 🟡 1 | 🟡 2 | 时间掩码数量 |
| **Train** | max_epoch | 2 | 🟡 **20** | 🟡 **10** | 更多epoch用于微调 |
| | use_deepspeed | true | 🔴 **false** | 🔴 **false** | 关闭分布式训练 |
| | keep_nbest_models | 200 | 🟡 **10** | 🟡 **5** | 节省磁盘空间 |
| | log_interval | 100 | 🟡 **10** | 🟡 **10** | 更频繁的日志 |
| | validate_interval | 2000 | ⚪ 无 | 🟡 **1000** | Stage1 更频繁验证 |
| | save_checkpoint_interval | 2000 | ⚪ 无 | 🟡 **1000** | Stage1 更频繁保存 |
| **Optimizer** | optim | adamw | 🟢 adamw | 🟢 adamw | 保持一致 |
| | lr | **5e-6** | 🔴 **5e-5** | 🔴 **1e-4** | 🔴 10-20倍提升 |
| | weight_decay | 0.0 | 🟡 **0.01** | 🟡 **0.01** | 新增正则化 |
| **Scheduler** | type | warmuplr | 🟢 warmuplr | 🟢 warmuplr | 保持一致 |
| | warmup_steps | 2500 | 🟡 **1000** | 🟡 **1000** | 更短预热期 |
| **Dataset** | type | FunASR | 🟢 FunASR | 🟢 FunASR | 保持一致 |
| | batch_size | 6000 | 🟢 6000 | 🟢 6000 | Token级别批处理 |
| | max_token_length | 3500 | 🟡 **1024** | 🟢 3500 | 内存限制(config_8k) |
| | audio_encoder_downsample_rate | 6 | 🟡 **2** | 🟡 **2** | 降采样率调整 |
| | num_workers | 4 | 🟢 4 | 🟡 **8** | Stage1 更多工作进程 |
| | data_split_num | 256 | 🟡 **512** | 🟢 256 | 数据分割数 |
| | batch_size_sample_max | 10 | 🟡 **15** | 🟢 10 | 采样批大小 |
| | retry | 2000 | 🟡 **20** | 🟢 2000 | 重试次数 |
| **Tokenizer** | type | HuggingfaceTokenizer | 🟢 HuggingfaceTokenizer | 🟢 HuggingfaceTokenizer | 保持一致 |
| | init_param_path | ${llm_conf.init_param_path} | 🟡 Qwen3-0.6B | 🟡 Qwen3-0.6B | 显式指定 |
| | unk_symbol | ⚪ 无 | 🟠 `<unk>` | 🟠 `<unk>` | 新增 |

---

## 分类统计

### 🔴 关键修改 (CRITICAL)
这些修改直接影响模型训练行为:

```
1. audio_encoder_conf.freeze:      true → false (编码器变为可训练)
2. optim_conf.lr:                   5e-6 → 5e-5 或 1e-4 (学习率提升10-20倍)
3. train_conf.use_deepspeed:        true → false (关闭分布式训练)
4. ctc_decoder_conf.freeze (stage1): false → true (Stage1冻结CTC)
5. frontend_conf.fs (stage1):       16000 → 8000 (Stage1使用原生采样率)
```

### 🟡 重要修改 (MODIFIED)
这些修改优化微调性能和资源使用:

```
- max_epoch: 2 → 10-20 (增加训练轮数)
- keep_nbest_models: 200 → 5-10 (节省磁盘)
- log_interval: 100 → 10 (更频繁日志)
- weight_decay: 0.0 → 0.01 (加入正则化)
- warmup_steps: 2500 → 1000 (缩短预热期)
- max_token_length: 3500 → 1024 (内存限制)
- time_mask_width_range: 新增 [0,10] 或 [0,50]
```

### 🟠 新增参数 (NEW)
预训练模型中不存在,微调配置中新增:

```
- specaug: SpecAugLFR (数据增强)
- specaug_conf.* (所有SpecAug配置)
- tokenizer_conf.unk_symbol: <unk>
```

### ⚪ 省略参数 (OMITTED)
未在微调配置中指定,使用代码默认值:

```
- audio_encoder_conf: attention_heads, linear_units, num_blocks, tp_blocks
- audio_encoder_conf: dropout_rate, positional_dropout_rate, attention_dropout_rate
- audio_encoder_conf: input_layer, pos_enc_class, normalize_before, kernel_size
- audio_encoder_conf: sanm_shfit, selfattention_layer_type, feat_permute
- llm_conf: hub, llm_dtype, init_param_path (使用pretrained default)
- llm_conf.lora_conf: (完全使用default)
- ctc_decoder_conf: downsample_rate, ffn_dim, llm_dim, encoder_dim, n_layer
- frontend_conf: window, frame_length, frame_shift, lfr_m, lfr_n, cmvn_file
- 其他所有未明确列出的参数
```

---

## 按用途分类

### 模型架构 - 冻结/可训练
| 组件 | 预训练 | 微调 | 状态 |
|------|------|-----|------|
| Audio Encoder | ❌ 冻结 | ✅ **可训练** | 🔴 **改变** |
| LLM | ❌ 冻结 | ❌ 冻结 | ✅ 保持 |
| Audio Adaptor | ❌ 冻结 | ❌ 冻结 | ✅ 保持 |
| CTC Decoder | ✅ 可训练 | ⚠️ 按Stage | 🔴 **改变** |

### 优化器与学习率
| 参数 | 预训练 | 微调 | 解释 |
|------|------|-----|------|
| 优化器 | AdamW | AdamW | 一致 |
| 学习率 | 5e-6 | 5e-5 ~ 1e-4 | 🔴 **提升10-20倍** |
| 权重衰减 | 0.0 | 0.01 | 🟡 新增正则化 |
| 梯度累积 | 1 | 1 | 一致 |

### 训练策略
| 参数 | 预训练 | 微调 | 原因 |
|------|------|-----|------|
| Epoch | 2 | 10-20 | 微调需要更多epoch |
| 预热步数 | 2500 | 1000 | 微调预热期缩短 |
| DeepSpeed | 启用 | 禁用 | 单GPU微调 |
| 保存模型数 | 200 | 5-10 | 节省磁盘空间 |

### 数据处理
| 参数 | 预训练 | 微调 | 说明 |
|------|------|-----|------|
| 数据增强 | 无 | SpecAugLFR | 🟠 新增 |
| 频率掩码 | 无 | [0,10-15] | 8kHz适配 |
| 时间掩码 | 无 | [0,10-50] | VoIP丢帧模拟 |
| 最大序列长 | 3500 | 1024* | *config_8k_telephone受内存限制 |
| 降采样率 | 6 | 2 | 降低频率分辨率 |

---

## 配置使用指南

### 使用 `config_8k_telephone.yaml` 时机
✅ 需要:
- 完整的encoder + CTC decoder微调
- 8kHz音频通过上采样到16kHz
- 较低的学习率(5e-5)支持多组件训练
- 内存受限的单GPU环境

❌ 不适合:
- 需要大序列长度(max_token_length限制为1024)
- 需要保留所有checkpoint(只保留最佳10个)

### 使用 `stage1_encoder_adapt.yaml` 时机
✅ 需要:
- 专注encoder适应8kHz频谱
- 更高学习率(1e-4)快速适应
- 更高的时间掩码([0,50])模拟VoIP丢帧
- 原生8kHz处理(不上采样)

❌ 不适合:
- 需要同时训练CTC decoder
- 内存充足可用更长序列

### 使用 `config_8k_telephone_from_hub.yaml` 时机
✅ 从ModelScope Hub加载预训练模型
❌ 当已有本地模型权重时

---

## 配置验证清单

在开始训练前检查:

- [ ] **Encoder是否可训练?**
  - config_8k_telephone: `audio_encoder_conf.freeze: false` ✓
  - stage1_encoder_adapt: `audio_encoder_conf.freeze: false` ✓

- [ ] **LLM和Adaptor是否冻结?**
  - `llm_conf.freeze: true` ✓
  - `audio_adaptor_conf.freeze: true` ✓

- [ ] **学习率是否合理?**
  - config_8k_telephone: `5e-5` (安全起点)
  - stage1_encoder_adapt: `1e-4` (encoder-only训练)

- [ ] **DeepSpeed是否禁用?**
  - `train_conf.use_deepspeed: false` ✓

- [ ] **SpecAug是否启用?**
  - `specaug: SpecAugLFR` ✓

- [ ] **预训练权重是否加载?**
  - 使用 `init_param=path/to/FunAudioLLM/Fun-ASR-Nano-2512/model.pt`

- [ ] **数据路径是否正确?**
  - `train_data_set_list` 指向实际数据

- [ ] **输出目录是否可写?**
  - `output_dir` 有足够空间保存5-10个checkpoint

---

## 预期训练效果

| 指标 | 预期值 | 说明 |
|------|-------|------|
| 初始Loss | ~2-3 | 从预训练模型开始 |
| Loss下降 | 30-50% | 正常微调效果 |
| 最佳Epoch | 5-15 | 通常中间阶段收敛 |
| 训练时间 | 按数据量 | 取决于GPU和数据规模 |
| 推理延迟 | 无增加 | 模型大小不变 |

---

## 常见问题

**Q: 为什么要冻结Encoder再解冻?**  
A: 初始化更稳定。如果直接用高LR训练encoder,可能破坏预训练权重。

**Q: stage1为什么使用8kHz而main config用16kHz?**  
A: 不同策略。stage1专注encoder频谱适应;main config需要兼容16kHz frontend。

**Q: 学习率5e-5太小了?**  
A: 相对预训练的5e-6已经提升10倍。微调通常需要保守的学习率。

**Q: 能否使用更长的max_token_length?**  
A: 可以,但需要更大GPU显存。1024是内存平衡点。

**Q: CTC Decoder为什么在stage1冻结?**  
A: 让encoder适应8kHz后再训练CTC,分阶段更稳定。

