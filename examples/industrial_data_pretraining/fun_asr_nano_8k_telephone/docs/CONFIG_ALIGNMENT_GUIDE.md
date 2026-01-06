# 配置对齐指南 - Fun-ASR-Nano 8kHz电话频道微调

## 📚 文档导航

本目录包含三份配置对齐文档,用于理解预训练模型配置与微调配置的差异:

### 1. **QUICK_CONFIG_DIFF.md** ⭐ (推荐首先阅读)
快速参考指南,展示最关键的配置差异。
- **适合:** 想快速了解改了什么
- **内容:** 关键参数对比、学习率、冻结策略
- **阅读时间:** 5-10分钟
- **使用场景:** 修改配置前快速检查清单

### 2. **CONFIG_COMPARISON_TABLE.md** 📊 (表格参考)
详细的表格对比,按模块分类显示所有差异。
- **适合:** 需要查看特定模块的改动
- **内容:** 完整的参数对比表、分类统计、验证清单
- **阅读时间:** 15-20分钟
- **使用场景:** 调试特定参数时查阅

### 3. **CONFIGURATION_ALIGNMENT.md** 📖 (深度参考)
最详细的配置文档,包含每个参数的完整说明。
- **适合:** 深度理解每个参数的作用和改动原因
- **内容:** 完整的参数映射、代码默认值、修改原因详解
- **阅读时间:** 30-40分钟
- **使用场景:** 学习或定制配置时的参考手册

---

## 🎯 按场景选择文档

### 场景1: 我想快速开始训练
```
1. 阅读 QUICK_CONFIG_DIFF.md 第 1-3 节 (5分钟)
   ↓ 了解Encoder、学习率、CTC的改动
   ↓
2. 检查 CONFIG_COMPARISON_TABLE.md 的"配置验证清单" (2分钟)
   ↓
3. 运行训练脚本
```

### 场景2: 我需要修改学习率或超参数
```
1. 查看 QUICK_CONFIG_DIFF.md 第 3 节 (学习率说明) (2分钟)
   ↓
2. 查看 CONFIG_COMPARISON_TABLE.md 的"优化器与学习率"表 (2分钟)
   ↓
3. 根据你的数据规模调整参数
   ↓
4. 使用 CONFIGURATION_ALIGNMENT.md 验证是否有关联参数需要调整
```

### 场景3: 我想理解Stage1和Main config的区别
```
1. 阅读 QUICK_CONFIG_DIFF.md 第 10 节 (2分钟)
   ↓ "参数源按阶段"部分明确说明了两个版本的区别
   ↓
2. 对比 stage1_encoder_adapt.yaml 和 config_8k_telephone.yaml
   ↓
3. 查看 CONFIG_COMPARISON_TABLE.md 的CTC Decoder和Frontend行
```

### 场景4: 我需要调试训练问题
```
1. 快速浏览 QUICK_CONFIG_DIFF.md 的"关键参数"部分 (3分钟)
   ↓
2. 查看 CONFIG_COMPARISON_TABLE.md 的"预期训练效果"表
   ↓
3. 根据问题查看对应的参数:
   - Loss不下降 → 检查学习率、weight_decay
   - 显存不足 → 检查max_token_length、batch_size
   - 训练太慢 → 检查num_workers、log_interval
   ↓
4. 用 CONFIGURATION_ALIGNMENT.md 查看详细说明
```

### 场景5: 我想定制化配置
```
1. 完整阅读 QUICK_CONFIG_DIFF.md (15分钟)
   ↓ 理解所有关键改动
   ↓
2. 系统阅读 CONFIG_COMPARISON_TABLE.md (20分钟)
   ↓ 了解完整的参数映射
   ↓
3. 参考 CONFIGURATION_ALIGNMENT.md (20分钟)
   ↓ 理解每个参数的作用
   ↓
4. 创建自己的config.yaml
```

---

## 🔑 关键概念速查表

### 什么被改了? (🔴 CRITICAL CHANGES)

| 参数 | 预训练 | 微调 | 为什么? |
|------|------|-----|--------|
| `audio_encoder_conf.freeze` | `true` | `false` | 需要适应8kHz频谱 |
| `optim_conf.lr` | `5e-6` | `5e-5 ~ 1e-4` | 预训练LR太小,微调需要更高LR |
| `train_conf.use_deepspeed` | `true` | `false` | 关闭分布式,使用单GPU |
| `ctc_decoder_conf.freeze` | `false` | Stage1:`true`, Main:`false` | Stage1专注encoder,Main训练CTC |
| `frontend_conf.fs` | `16000` | Stage1:`8000`, Main:`16000` | Stage1原生8kHz,Main上采样 |

### 什么没变? (🟢 ALIGNED)

| 组件 | 状态 |
|------|------|
| LLM (Qwen3-0.6B) | ✅ 始终冻结 |
| Audio Adaptor | ✅ 始终冻结 |
| Model类型 | ✅ 始终FunASRNano |
| Batch大小 | ✅ 始终6000 tokens |

### 什么是新增的? (🟠 NEW)

| 项目 | 预训练中 | 微调中 |
|------|---------|--------|
| SpecAugmentation | ❌ 无 | ✅ 新增SpecAugLFR |
| 频率掩码 | ❌ 无 | ✅ 新增[0,10-15] |
| 时间掩码 | ❌ 无 | ✅ 新增[0,10-50] |
| Weight Decay | ❌ 0.0 | ✅ 0.01 |

### 什么使用默认值? (⚪ OMITTED)

许多参数未在微调配置中指定,使用代码默认值:
- Encoder参数: `attention_heads`, `linear_units`, `num_blocks` 等
- LLM参数: `llm_dtype`, `init_param_path` 等
- CTC参数: `ffn_dim`, `encoder_dim` 等

**关键:** 这些都来自预训练模型,应该保持不变。

---

## ⚠️ 常见错误和纠正

### ❌ 错误1: 学习率太小
```yaml
# 错误
optim_conf:
  lr: 5e-6  # 这是预训练的LR,不适合微调
  
# 正确 (config_8k_telephone)
optim_conf:
  lr: 5e-5  # 或 1e-4 for stage1
```

### ❌ 错误2: 忘记解冻Encoder
```yaml
# 错误
audio_encoder_conf:
  freeze: true  # Encoder未训练,无法适应8kHz

# 正确
audio_encoder_conf:
  freeze: false  # Encoder可以训练
```

### ❌ 错误3: 启用DeepSpeed单机训练
```yaml
# 错误
train_conf:
  use_deepspeed: true  # 单GPU不需要DeepSpeed

# 正确
train_conf:
  use_deepspeed: false  # 关闭分布式训练
```

### ❌ 错误4: 没有Weight Decay
```yaml
# 错误
optim_conf:
  weight_decay: 0.0  # 微调容易过拟合

# 正确
optim_conf:
  weight_decay: 0.01  # 加入L2正则化
```

### ❌ 错误5: 没有SpecAugmentation
```yaml
# 错误
# (config中没有specaug部分)

# 正确
specaug: SpecAugLFR
specaug_conf:
  apply_freq_mask: true
  freq_mask_width_range: [0, 15]
  apply_time_mask: true
  time_mask_width_range: [0, 10]  # 或 [0, 50] for stage1
  num_time_mask: 1  # 或 2 for stage1
```

---

## 📋 训练前检查清单

复制下面的清单,在开始训练前完成检查:

```
[ ] 1. Encoder是否可训练?
    - audio_encoder_conf.freeze: false
    [ ] Yes [ ] No

[ ] 2. LLM和Adaptor是否冻结?
    - llm_conf.freeze: true
    - audio_adaptor_conf.freeze: true
    [ ] Yes [ ] No

[ ] 3. 学习率是否合理?
    - config_8k_telephone: lr: 5e-5
    - stage1_encoder_adapt: lr: 1e-4
    [ ] Yes [ ] No (自定义) [ ] Need Review

[ ] 4. DeepSpeed是否禁用?
    - train_conf.use_deepspeed: false
    [ ] Yes [ ] No

[ ] 5. SpecAugmentation是否启用?
    - specaug: SpecAugLFR
    [ ] Yes [ ] No

[ ] 6. 权重衰减是否启用?
    - optim_conf.weight_decay: 0.01
    [ ] Yes [ ] No

[ ] 7. CTC Decoder状态是否正确?
    - config_8k_telephone: freeze: false (训练)
    - stage1_encoder_adapt: freeze: true (冻结)
    [ ] Yes [ ] No [ ] Other: ____

[ ] 8. Frontend采样率是否正确?
    - config_8k_telephone: fs: 16000
    - stage1_encoder_adapt: fs: 8000
    [ ] Verified [ ] Need Check

[ ] 9. 预训练权重是否加载?
    - 使用 --init_param=path/to/model.pt
    [ ] Yes [ ] No [ ] Will set in script

[ ] 10. 输出目录是否准备好?
    - output_dir: 有至少50GB可用空间
    [ ] Yes [ ] No [ ] Need Fix

已检查项: _____ / 10
建议: 所有项都应该勾选"Yes"或"Verified"
```

---

## 🚀 快速启动

### 使用默认配置
```bash
# 使用 config_8k_telephone.yaml (推荐初学者)
cd /workspace/share/LLMFunASR
python -m funasr.bin.train_asr \
  --config examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/conf/config_8k_telephone.yaml \
  --init_param <path-to-pretrained-model.pt> \
  --train_data_set_list <your-training-data> \
  --valid_data_set_list <your-validation-data> \
  --output_dir <output-directory>
```

### 使用Stage1配置
```bash
# 首先使用 stage1_encoder_adapt.yaml 适应8kHz
python -m funasr.bin.train_asr \
  --config examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/conf/stage1_encoder_adapt.yaml \
  --init_param <path-to-pretrained-model.pt> \
  --train_data_set_list <your-training-data> \
  --valid_data_set_list <your-validation-data> \
  --output_dir <stage1-output>
```

### 自定义超参数
```bash
# 在命令行覆盖配置
python -m funasr.bin.train_asr \
  --config examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/conf/config_8k_telephone.yaml \
  --optim_conf.lr 1e-4 \
  --train_conf.max_epoch 15 \
  --train_conf.keep_nbest_models 5 \
  --init_param <path> \
  --train_data_set_list <data> \
  --output_dir <output>
```

---

## 📖 深入学习资源

### 理解预训练 vs 微调的区别
- 预训练配置: `/gpfs01/nfs_share/finrc/liangguang/cache/modelscope/models/FunAudioLLM/Fun-ASR-Nano-2512/config.yaml`
- 对比原始配置以理解全部改动

### 参考配置文件
- `config_8k_telephone.yaml` - 完整微调 (推荐)
- `stage1_encoder_adapt.yaml` - Encoder适应专用
- `config_8k_telephone_from_hub.yaml` - 从Hub加载模型

### 理解数据和SpecAug
- 查看 `QUICK_CONFIG_DIFF.md` 第 7️⃣ 节 (SpecAugmentation)
- 了解为什么8kHz需要特殊的时间掩码配置

### 学习率调整
- 小数据 (<1000h): 保守的学习率(5e-5)
- 大数据 (>1000h): 可以尝试更高的学习率(1e-4)
- 监控loss曲线: 应该平稳下降,不应该发散

---

## 💬 常见问题

**Q: 我应该使用哪个配置?**
A: 推荐用 `config_8k_telephone.yaml` 开始。只有在需要特别适应8kHz时才用 `stage1_encoder_adapt.yaml`。

**Q: 学习率5e-5合适吗?**
A: 如果loss不下降,可以试试1e-4。如果训练不稳定,降低到2e-5。

**Q: 为什么Stage1使用8kHz而Main使用16kHz?**
A: 不同策略。Stage1让encoder学习8kHz特性;Main用上采样使用标准16kHz处理链。

**Q: 能否同时训练多个组件?**
A: 可以,但需要更保守的学习率。不同组件需要平衡。

**Q: 如何知道训练是否成功?**
A: 查看 `CONFIG_COMPARISON_TABLE.md` 的"预期训练效果"表。Loss应下降30-50%。

**Q: 可以跳过SpecAugmentation吗?**
A: 不推荐。SpecAug是微调成功的关键,尤其对8kHz数据。

---

## 📞 反馈和改进

如果发现配置对齐文档有问题或可以改进,请:
1. 检查 CONFIGURATION_ALIGNMENT.md 是否有相同问题
2. 检查原始预训练配置是否有变更
3. 提交问题报告

---

**最后更新:** 2025-01-05  
**配置参考:** Fun-ASR-Nano-2512 (Fun-ASR-Nano-2512/config.yaml)  
**文档版本:** 1.0  
