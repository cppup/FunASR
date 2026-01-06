# 🔧 配置对齐文档说明

## 📌 快速导航

本实验的所有配置都是基于预训练模型 **Fun-ASR-Nano-2512** 进行微调的。为了明确哪些参数被修改了、哪些保持不变、哪些使用默认值，我们提供了以下4份对齐文档：

### 1. **CONFIG_ALIGNMENT_GUIDE.md** 📖 [START HERE]
👉 **首先阅读这个文件** - 它指导你如何使用其他三份文档
- 按场景提供快速导航
- 关键概念速查表
- 常见错误与纠正
- 训练前检查清单

### 2. **QUICK_CONFIG_DIFF.md** ⚡ [5-10分钟]
最快速的参考 - 展示所有关键改动和差异
- 🔴 关键改动 (CRITICAL)
- 🟡 重要修改 (MODIFIED)
- 🟠 新增参数 (NEW)
- ⚪ 省略参数 (OMITTED)
- 适合: 快速检查修改了什么

### 3. **CONFIG_COMPARISON_TABLE.md** 📊 [15-20分钟]
完整的表格对比视图，按模块分类
- 模块化的参数对比表
- 按用途分类的参数汇总
- 预期训练效果
- 适合: 查看特定模块或参数

### 4. **CONFIGURATION_ALIGNMENT.md** 📚 [30-40分钟深度参考]
最详细的完整文档，包含每个参数的完整信息
- 所有参数的完整映射
- 代码默认值说明
- 每个修改的原因解释
- 适合: 深度学习和定制配置

---

## 🎯 按场景快速选择

| 我想要... | 推荐文档 | 时间 |
|---------|--------|------|
| 快速了解什么改了 | QUICK_CONFIG_DIFF | 5分钟 |
| 查询特定参数 | CONFIG_COMPARISON_TABLE | 2分钟 |
| 理解参数的详细原因 | CONFIGURATION_ALIGNMENT | 10分钟 |
| 修改学习率或超参数 | CONFIG_COMPARISON_TABLE → QUICK_CONFIG_DIFF | 5分钟 |
| 调试训练问题 | QUICK_CONFIG_DIFF + CONFIG_COMPARISON_TABLE | 10分钟 |
| 完整理解整个配置 | 按顺序读所有4个文档 | 1小时 |
| 训练前最后检查 | CONFIG_ALIGNMENT_GUIDE 的检查清单 | 5分钟 |

---

## 🔑 关键改动一览

### 最重要的3个改动：

**1️⃣ Audio Encoder现在可训练** 🔴
```yaml
# 预训练: freeze: true (冻结)
# 微调:   freeze: false (可训练) ← 关键改动!
```

**2️⃣ 学习率提升10-20倍** 🔴
```yaml
# 预训练: lr: 5e-6
# 微调:   lr: 5e-5 (main) 或 1e-4 (stage1) ← 关键改动!
```

**3️⃣ DeepSpeed关闭** 🔴
```yaml
# 预训练: use_deepspeed: true (分布式)
# 微调:   use_deepspeed: false (单GPU) ← 关键改动!
```

其他常见修改：
- 🟡 Weight decay从0增加到0.01 (防止过拟合)
- 🟡 SpecAugmentation新增 (数据增强)
- 🟡 max_epoch从2增加到10-20 (更多训练轮数)
- 🟡 DeepSpeed关闭 (单GPU友好)

---

## ✅ 训练前7步快速检查

```bash
# Step 1: 检查Encoder是否可训练
grep -A2 "audio_encoder_conf:" conf/config_8k_telephone.yaml | grep freeze

# Step 2: 检查学习率
grep "lr:" conf/config_8k_telephone.yaml

# Step 3: 检查DeepSpeed
grep "use_deepspeed:" conf/config_8k_telephone.yaml

# Step 4: 检查Weight Decay
grep "weight_decay:" conf/config_8k_telephone.yaml

# Step 5: 检查SpecAug
grep "specaug:" conf/config_8k_telephone.yaml

# Step 6: 检查LLM冻结
grep -A3 "llm_conf:" conf/config_8k_telephone.yaml | grep freeze

# Step 7: 检查预训练权重路径
echo "记住用 --init_param 加载预训练模型"
```

预期输出应该是：
```
✅ audio_encoder freeze: false
✅ lr: 5e-5 (或 1e-4)
✅ use_deepspeed: false
✅ weight_decay: 0.01
✅ specaug: SpecAugLFR
✅ llm freeze: true
✅ 使用 --init_param 加载模型.pt
```

---

## 🗂️ 配置文件说明

本目录的 `conf/` 下有多个配置文件：

| 文件 | 用途 | 推荐 |
|------|------|------|
| `config_8k_telephone.yaml` | **完整微调** (Encoder + CTC) | ⭐⭐⭐ 首选 |
| `stage1_encoder_adapt.yaml` | 只微调Encoder (8kHz适应) | ⭐⭐ 专用 |
| `stage2_adapter_align.yaml` | 微调Adaptor | ⭐ 高级 |
| `stage3_lora_domain.yaml` | 使用LoRA微调 | ⭐ 高级 |
| `config_8k_telephone_from_hub.yaml` | 从Hub下载模型 | ⭐⭐ 备选 |

**推荐流程**: 
- 初学者: 用 `config_8k_telephone.yaml`
- 需要8kHz优化: 用 `stage1_encoder_adapt.yaml`
- 高级定制: 参考 `stage2/3` 创建自己的配置

---

## 📊 配置对比矩阵

| 参数 | 预训练 | config_8k_telephone | stage1 | 说明 |
|------|------|-----------------|--------|------|
| `audio_encoder.freeze` | `true` | `false` ⚠️ | `false` ⚠️ | Encoder可训练 |
| `optim_conf.lr` | `5e-6` | `5e-5` ⚠️ | `1e-4` ⚠️ | 提升10-20倍 |
| `train_conf.max_epoch` | `2` | `20` | `10` | 更多训练轮数 |
| `train_conf.use_deepspeed` | `true` | `false` | `false` | 单GPU训练 |
| `optim_conf.weight_decay` | `0.0` | `0.01` | `0.01` | 新增正则化 |
| `specaug` | ❌ | ✅ | ✅ | 新增数据增强 |
| `llm_conf.freeze` | `true` | `true` | `true` | 保持冻结 |
| `ctc_decoder.freeze` | `false` | `false` | `true` | Stage1冻结 |

⚠️ = 关键改动

---

## 🚀 立即开始

### 最快5分钟启动训练

```bash
cd /workspace/share/LLMFunASR

# 1. 读一遍QUICK_CONFIG_DIFF (5分钟)
cat examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/QUICK_CONFIG_DIFF.md

# 2. 准备数据和输出目录
export DATA_DIR="/path/to/your/8k/data"
export OUTPUT_DIR="/path/to/output"

# 3. 启动训练
python -m funasr.bin.train_asr \
  --config examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/conf/config_8k_telephone.yaml \
  --init_param /gpfs01/nfs_share/finrc/liangguang/cache/modelscope/models/FunAudioLLM/Fun-ASR-Nano-2512/model.pt \
  --train_data_set_list $DATA_DIR/train.txt \
  --valid_data_set_list $DATA_DIR/valid.txt \
  --output_dir $OUTPUT_DIR \
  --gpu_id 0
```

---

## ❓ 常见问题速查

**Q: 学习率5e-5是否太小?**  
A: 对于预训练模型微调是合适的。如果loss不下降可以试试1e-4。

**Q: 可以跳过SpecAugmentation吗?**  
A: 不建议。SpecAug是8kHz微调的关键部分。

**Q: stage1和main config哪个好?**  
A: 大多数情况用main config。stage1只在需要专门8kHz适应时用。

**Q: 为什么Stage1用8kHz而main用16kHz?**  
A: 不同策略。Stage1让encoder学习8kHz;main用上采样使用标准处理链。

**更多问题?** 参考 CONFIG_ALIGNMENT_GUIDE.md 的"常见问题"部分

---

## 📚 文件阅读顺序建议

### 🟢 快速上手 (15分钟)
1. 这个文件 (README_CONFIG.md) - 2分钟
2. QUICK_CONFIG_DIFF.md 前3节 - 5分钟
3. CONFIG_ALIGNMENT_GUIDE.md 的检查清单 - 5分钟
4. 开始训练 ✅

### 🟡 标准学习 (45分钟)
1. CONFIG_ALIGNMENT_GUIDE.md - 15分钟
2. QUICK_CONFIG_DIFF.md - 10分钟
3. CONFIG_COMPARISON_TABLE.md - 20分钟
4. 根据需要参考 CONFIGURATION_ALIGNMENT.md

### 🔴 深度学习 (2小时)
按顺序完整阅读：
1. CONFIG_ALIGNMENT_GUIDE.md
2. QUICK_CONFIG_DIFF.md
3. CONFIG_COMPARISON_TABLE.md
4. CONFIGURATION_ALIGNMENT.md
5. 对比实际的配置文件

---

## 📝 说明

- **创建日期**: 2025-01-05
- **参考模型**: Fun-ASR-Nano-2512
- **参考配置**: `/gpfs01/nfs_share/finrc/liangguang/cache/modelscope/models/FunAudioLLM/Fun-ASR-Nano-2512/config.yaml`
- **文档版本**: 1.0

---

**下一步**: 👉 打开 `QUICK_CONFIG_DIFF.md` 了解所有关键改动
