# Fun-ASR-Nano 训练数据准备完整指南

## 📋 概述

本文档汇总了针对 **Fun-ASR-Nano** 语音识别模型的完整训练数据准备方案。通过反推模型代码、配置文件和示例，我们已生成了详细的数据格式规范、修复工具和快速开始指南。

---

## 🎯 快速导航

### 新手入门（5分钟）
👉 **立即开始**：[QUICK_START_TRAINING.md](QUICK_START_TRAINING.md)
- 5步快速上手
- 完整工作流示例
- 常见问题解决

### 遇到问题
👉 **数据修复指南**：[FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md)
- train_data/val_data 问题诊断
- 四种修复方案
- 验证检查清单

### 深入了解
👉 **格式详解**：[DATA_FORMAT_ANALYSIS.md](DATA_FORMAT_ANALYSIS.md)
- JSONL 格式规范与示例
- 模型数据流完整说明
- 特征提取配置
- 数据融合机制

### 项目总结
👉 **项目总结**：[TRAINING_DATA_SUMMARY.md](TRAINING_DATA_SUMMARY.md)
- 核心发现与技术要点
- 生成资源总结
- 参考指令速查

---

## 📦 提供的工具

### 1. 数据转换脚本
**文件**：`prepare_training_data.py` (14KB)

**功能**：
- CSV → JSONL 格式转换
- 路径处理与验证
- 音频文件检查
- 数据有效性验证

**使用**：
```bash
# 从 CSV 转换
python prepare_training_data.py \
    --input data.csv \
    --output train.jsonl \
    --audio_dir /path/to/audio \
    --validate

# CSV 格式
# source,target
# audio/001.wav,转录文本
# audio/002.wav,第二条转录
```

**输出**：
```jsonl
{"key": "sample_000001", "source": "/abs/path/audio/001.wav", "target": "转录文本"}
```

### 2. 数据修复脚本
**文件**：`fix_training_data.py` (6.7KB)

**功能**：
- 修复 JSONL 格式错误
- 转换相对路径为绝对路径
- 验证音频文件存在性
- 清理文本内容
- 生成修复统计报告

**使用**：
```bash
# 修复有问题的 JSONL
python fix_training_data.py \
    broken.jsonl \
    fixed.jsonl \
    --audio_dir /path/to/audio \
    --verbose

# 输出示例
# ============================================================
# REPAIR STATISTICS
# ============================================================
# Total records:              1000
# Successfully fixed:         980
#   Invalid JSON:             5
#   Missing audio files:      10
#   Empty text:               5
# ============================================================
# Success rate: 98.0%
```

### 3. 数据准备示例
**文件**：`examples/prepare_data_example.py` (5.4KB)

**功能**：
- 演示手动数据准备
- 生成示例 JSONL
- 详细注释和说明

**使用**：
```bash
python examples/prepare_data_example.py
# 生成示例 train.jsonl, val.jsonl
```

---

## 📄 标准数据格式

### JSONL（推荐格式）

```jsonl
{
  "key": "sample_000001",
  "source": "/abs/path/to/audio.wav",
  "source_len": 10,
  "target": "这是转录文本",
  "target_len": 5,
  "domain": "telephone",
  "speaker_id": "speaker_001"
}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| source | string | ✓ | 音频文件**绝对路径** |
| target | string | ✓ | UTF-8 转录文本 |
| key | string | × | 唯一标识符（自动生成） |
| source_len | int | × | 音频时长（秒） |
| target_len | int | × | 文本长度（字符） |
| domain | string | × | 领域标记 |
| speaker_id | string | × | 说话人 ID |

### 音频格式
- **采样率**：16000 Hz
- **声道**：单声道
- **格式**：WAV、MP3、FLAC
- **编码**：UTF-8

### 文本格式
- **编码**：UTF-8
- **语言**：中文、英文或混合
- **内容**：清晰自然语音转录
- **避免**：特殊符号（[笑]、[停顿]等）

---

## 🚀 三步快速开始

### 步骤 1：准备数据

**选择 A - 从 CSV 转换**（推荐）：
```bash
python prepare_training_data.py \
    --input data.csv \
    --output train.jsonl \
    --audio_dir /path/to/audio
```

**选择 B - 修复现有 JSONL**：
```bash
python fix_training_data.py \
    old_train.jsonl \
    train.jsonl \
    --audio_dir /path/to/audio
```

**选择 C - 手动创建**：
```bash
python examples/prepare_data_example.py
# 编辑生成的文件，添加自己的数据
```

### 步骤 2：验证数据

```bash
# 快速验证
python -c "
import json, os
valid = 0
for line in open('train.jsonl'):
    if line.strip():
        r = json.loads(line)
        if os.path.isfile(r['source']):
            valid += 1
print(f'Valid records: {valid}')
"
```

### 步骤 3：启动训练

```bash
# 进入训练目录
cd examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# 更新配置文件中的数据路径
# vim finetune_8k_telephone.sh

# 启动训练
bash finetune_8k_telephone.sh
```

---

## 🔍 常见问题速查

### Q1: 路径错误
```
❌ {"source": "audio/001.wav"}  # 相对路径
✓ {"source": "/data/audio/001.wav"}  # 绝对路径

# 修复：使用 fix_training_data.py
python fix_training_data.py input.jsonl output.jsonl --audio_dir /data
```

### Q2: 文件不存在
```bash
# 检查文件
ls -la /data/audio/*.wav

# 检查 JSONL 中的路径
python -c "
import json, os
for i, line in enumerate(open('train.jsonl')):
    if line.strip():
        r = json.loads(line)
        if not os.path.isfile(r['source']):
            print(f'Line {i}: {r[\"source\"]} not found')
            if i >= 5: break
"
```

### Q3: JSONL 格式错误
```bash
# 验证 JSON 有效性
python -c "
import json
for i, line in enumerate(open('train.jsonl')):
    if line.strip():
        try:
            json.loads(line)
        except:
            print(f'Line {i}: Invalid JSON')
"

# 修复：使用 fix_training_data.py
python fix_training_data.py broken.jsonl fixed.jsonl
```

### Q4: 文本编码问题
```bash
# 确保 UTF-8 编码
file -i train.jsonl
# 应显示：charset=utf-8

# 转换编码
iconv -f GBK -t UTF-8 old.jsonl > new.jsonl
```

### Q5: 更多问题
👉 详见：[FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md)

---

## 📊 数据规模建议

| 数据规模 | 适用场景 | 学习率 | Epoch | 耗时 |
|---------|---------|--------|--------|------|
| 5-10h | 小样本微调 | 5e-5 | 20-30 | 1-2 天 |
| 10-50h | 标准微调 | 5e-5 | 15-20 | 2-5 天 |
| 50-100h | 中规模 | 5e-5 | 10-15 | 5-10 天 |
| >100h | 大规模 | 5e-5 | 5-10 | 10+ 天 |
| <5h | LoRA 微调 | 5e-4 | 50+ | 2-4 小时 |

---

## 📁 文件结构

```
/workspace/share/LLMFunASR/
│
├── 📖 文档
│   ├── README_DATA_PREPARATION.md      ← 本文件（导航页）
│   ├── QUICK_START_TRAINING.md         ← 5步快速开始
│   ├── DATA_FORMAT_ANALYSIS.md         ← 详细格式说明
│   ├── FIX_TRAINING_DATA_GUIDE.md      ← 问题修复指南
│   └── TRAINING_DATA_SUMMARY.md        ← 项目总结
│
├── 🐍 脚本
│   ├── prepare_training_data.py        ← CSV → JSONL 转换
│   ├── fix_training_data.py            ← JSONL 修复工具
│   └── examples/
│       └── prepare_data_example.py     ← 数据准备示例
│
└── 🎓 参考代码
    └── examples/industrial_data_pretraining/
        ├── fun_asr_nano/model.py       ← 模型代码（反推来源）
        └── fun_asr_nano_8k_telephone/  ← 完整训练示例
            ├── finetune_8k_telephone.sh
            ├── data/data_simulation.py
            └── conf/config_8k_telephone.yaml
```

---

## ⚙️ 工作流总结

```
准备原始数据
   ↓
选择转换方式 ─→ CSV 转换 / 手动创建 / 修复现有
   ↓
生成 JSONL 文件（train.jsonl, val.jsonl）
   ↓
验证数据完整性 ─→ 检查格式、文件、文本
   ↓
（可选）数据模拟 ─→ 8kHz 电话场景
   ↓
配置训练参数 ─→ learning_rate, batch_size, epochs
   ↓
启动训练 ─→ bash finetune_8k_telephone.sh
   ↓
推理评估 ─→ 计算 CER/WER 指标
```

---

## 💡 最佳实践

### 1. 使用绝对路径
```python
import os
path = os.path.abspath(relative_path)
```

### 2. 验证数据完整性
```python
assert os.path.isfile(record['source'])
assert record['target'].strip()
```

### 3. 标准化文本
```python
text = record['target'].strip()
text = ' '.join(text.split())  # 去重空格
```

### 4. 分离训练/验证集
```bash
# 80% 训练，20% 验证
split=$(($(wc -l < data.jsonl) * 80 / 100))
head -n $split data.jsonl > train.jsonl
tail -n +$((split + 1)) data.jsonl > val.jsonl
```

### 5. 定期备份
```bash
cp train.jsonl train.jsonl.bak
cp val.jsonl val.jsonl.bak
```

---

## 🔗 参考链接

- **完整分析**：[DATA_FORMAT_ANALYSIS.md](DATA_FORMAT_ANALYSIS.md) (10.7KB)
- **修复指南**：[FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md) (12.7KB)
- **快速开始**：[QUICK_START_TRAINING.md](QUICK_START_TRAINING.md) (8.1KB)
- **项目总结**：[TRAINING_DATA_SUMMARY.md](TRAINING_DATA_SUMMARY.md) (9.4KB)

---

## 🎓 关键技术要点

### 数据融合机制
- 音频特征通过 `fbank_beg` 定位插入位置
- 每个音频对应的 token 数由 `fake_token_len` 确定
- 支持多轮对话中多个音频

### 特征提取
- WavFrontend: 16kHz, 80 Mel channel, 25ms frame
- 自动重采样至 16kHz
- 支持多种音频格式

### 模型架构
- **编码器**：SenseVoiceEncoderSmall (512 dim)
- **适配器**：Transformer (1024 dim)
- **LLM**：Qwen3-0.6B (1024 dim)

### 8kHz 电话模拟
- 下采样到 8kHz
- 带通滤波 (300-3400 Hz)
- G.711 编解码压扩
- 噪声添加 (SNR 15-25 dB)
- 上采样回 16kHz

---

## 📝 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2026-01-05 | 初始版本，包含完整分析和工具 |

---

## ❓ 获取帮助

遇到问题时：

1. **快速查询**：
   - 路径问题？→ [FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md#常见错误)
   - 格式问题？→ [DATA_FORMAT_ANALYSIS.md](DATA_FORMAT_ANALYSIS.md#数据格式规范)
   - 训练问题？→ [QUICK_START_TRAINING.md](QUICK_START_TRAINING.md#常见问题解决)

2. **运行脚本**：
   - 格式错误 → `python fix_training_data.py`
   - CSV 转换 → `python prepare_training_data.py`
   - 验证数据 → 检查清单脚本

3. **查看日志**：
   - 训练日志：`exp_output/train.log`
   - 数据处理日志：脚本输出信息

---

## 📢 重要提示

✅ **必须项**：
- [ ] source：绝对路径，文件存在
- [ ] target：UTF-8 编码，非空
- [ ] JSONL 格式：有效 JSON

⚠️ **注意事项**：
- 使用绝对路径避免加载错误
- 验证数据完整性后再启动训练
- 预留充足磁盘空间（特别是数据模拟）
- 设置合理的学习率和 batch_size

---

**准备好了吗？** 👉 [立即开始](QUICK_START_TRAINING.md)

---

**最后更新**：2026-01-05 10:02 UTC
