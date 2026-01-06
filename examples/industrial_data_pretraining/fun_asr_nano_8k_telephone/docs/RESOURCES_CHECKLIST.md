# 📋 Fun-ASR-Nano 数据准备资源清单

## ✅ 已生成资源总览

### 📚 文档（4 份）

| 文件 | 大小 | 内容 | 用途 |
|------|------|------|------|
| **README_DATA_PREPARATION.md** | 10KB | 导航页面、快速参考 | 入门必读 |
| **QUICK_START_TRAINING.md** | 8.1KB | 5步流程、参考表 | 快速上手 |
| **DATA_FORMAT_ANALYSIS.md** | 10.7KB | 深度技术分析 | 深入学习 |
| **FIX_TRAINING_DATA_GUIDE.md** | 12.7KB | 问题诊断、修复方案 | 遇到问题时 |
| **TRAINING_DATA_SUMMARY.md** | 9.4KB | 核心要点总结 | 知识回顾 |

### 🐍 Python 脚本（3 个）

| 脚本 | 大小 | 功能 | 使用场景 |
|------|------|------|---------|
| **prepare_training_data.py** | 13.4KB | CSV → JSONL 转换 | 有 CSV 数据源 |
| **fix_training_data.py** | 6.7KB | 修复 JSONL 格式问题 | 数据有错误 |
| **prepare_data_example.py** | 5.4KB | 数据准备演示 | 学习参考 |

---

## 🎯 快速指引

### 我是新手，想快速开始
1. 📖 阅读：[README_DATA_PREPARATION.md](README_DATA_PREPARATION.md) (3 分钟)
2. 📖 学习：[QUICK_START_TRAINING.md](QUICK_START_TRAINING.md) (5 分钟)
3. 🐍 运行：`python prepare_training_data.py` (1 分钟)

**总耗时**：约 10 分钟

### 我有现成的 CSV 文件
```bash
python prepare_training_data.py \
    --input my_data.csv \
    --output train.jsonl \
    --audio_dir /path/to/audio \
    --validate
```

### 我的 JSONL 有问题
```bash
python fix_training_data.py \
    broken.jsonl \
    fixed.jsonl \
    --audio_dir /path/to/audio \
    --verbose
```

### 我想深入理解格式
📖 阅读：[DATA_FORMAT_ANALYSIS.md](DATA_FORMAT_ANALYSIS.md)
- 完整的数据流说明
- 模型架构与数据要求
- 特征提取配置
- 对话格式支持

### 我遇到了错误
📖 参考：[FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md)
- 问题诊断方法
- 常见错误与解决
- 验证检查清单

---

## 📊 核心数据格式

### JSONL 示例
```jsonl
{
  "key": "sample_000001",
  "source": "/abs/path/to/audio.wav",
  "target": "转录文本",
  "source_len": 10,
  "target_len": 5
}
```

### 必填字段
- ✅ `source`：音频文件绝对路径
- ✅ `target`：UTF-8 转录文本

### 可选字段
- `key`：唯一标识符（自动生成）
- `source_len`：音频时长（秒）
- `target_len`：文本长度（字符）
- `domain`：数据领域标记
- `speaker_id`：说话人标识

---

## 🔧 工具使用流程

### 流程 A：从 CSV 转换

```
CSV 文件
  ↓
prepare_training_data.py
  ↓
JSONL 文件
  ↓
train.jsonl + val.jsonl
```

**命令**：
```bash
python prepare_training_data.py \
    --input data.csv \
    --output train.jsonl \
    --audio_dir /audio
```

**CSV 格式**：
```
source,target
audio/001.wav,转录内容
audio/002.wav,第二条转录
```

### 流程 B：修复现有 JSONL

```
JSONL 文件（有问题）
  ↓
fix_training_data.py
  ↓
修复后 JSONL
  ↓
可用于训练
```

**命令**：
```bash
python fix_training_data.py \
    old.jsonl \
    fixed.jsonl \
    --audio_dir /audio
```

### 流程 C：手动创建

```
数据准备
  ↓
prepare_data_example.py（参考）
  ↓
手动编写
  ↓
JSONL 文件
```

**参考**：
```bash
python examples/prepare_data_example.py
```

---

## ✨ 功能速查

### 需要什么... → 查看...

| 需求 | 文档 | 脚本 |
|------|------|------|
| 快速了解数据格式 | QUICK_START | - |
| CSV 转 JSONL | - | prepare_training_data.py |
| 修复 JSONL 错误 | FIX_GUIDE | fix_training_data.py |
| 学习技术细节 | DATA_FORMAT | - |
| 解决具体问题 | FIX_GUIDE | fix_training_data.py |
| 参考代码示例 | - | prepare_data_example.py |
| 查找训练参数 | QUICK_START | - |
| 理解数据流 | DATA_FORMAT | - |

---

## 🎓 学习路径

### 初级（1 小时）
1. ✅ README_DATA_PREPARATION.md（导航页）
2. ✅ QUICK_START_TRAINING.md（快速流程）
3. ✅ 运行 prepare_training_data.py

### 中级（2 小时）
1. ✅ 学习 JSONL 格式规范
2. ✅ 理解数据融合机制
3. ✅ 使用 fix_training_data.py 修复数据

### 高级（3+ 小时）
1. ✅ 深入 DATA_FORMAT_ANALYSIS.md
2. ✅ 理解模型架构与数据流
3. ✅ 掌握所有参数配置

---

## 📋 检查清单

### 准备数据阶段
- [ ] 准备音频文件（16kHz, mono, WAV/MP3）
- [ ] 准备转录文本（UTF-8, 清晰内容）
- [ ] 创建 CSV 或 JSONL 文件
- [ ] 验证数据格式有效性
- [ ] 分离训练/验证集

### 修复数据阶段（如需要）
- [ ] 运行 fix_training_data.py
- [ ] 检查修复统计报告
- [ ] 验证修复后的数据
- [ ] 确认所有文件都找到

### 启动训练前
- [ ] JSONL 格式有效
- [ ] 所有音频文件存在
- [ ] 文本编码正确（UTF-8）
- [ ] 分割训练/验证集
- [ ] 配置文件参数正确

---

## 🚀 三分钟快速体验

```bash
# 1. 创建示例数据
cd /workspace/share/LLMFunASR
python examples/prepare_data_example.py

# 2. 验证生成的文件
cat /workspace/data/funasr/train.jsonl | head -3

# 3. 查看帮助信息
python prepare_training_data.py --help
python fix_training_data.py --help
```

---

## 📞 常见问题速查

**Q: JSONL 格式错误怎么办？**
```bash
python fix_training_data.py broken.jsonl fixed.jsonl
```

**Q: 音频文件路径错误怎么办？**
```bash
python fix_training_data.py bad.jsonl good.jsonl --audio_dir /audio
```

**Q: 如何验证数据有效？**
```python
python -c "
import json, os
for line in open('train.jsonl'):
    r = json.loads(line)
    assert os.path.isfile(r['source']), f'Missing: {r[\"source\"]}'
print('✓ All files exist')
"
```

**Q: 如何转换编码？**
```bash
iconv -f GBK -t UTF-8 old.jsonl > new.jsonl
```

---

## 📈 资源大小统计

```
总大小：约 100 KB

文档：
  - README_DATA_PREPARATION.md        10 KB
  - QUICK_START_TRAINING.md           8.1 KB
  - DATA_FORMAT_ANALYSIS.md           10.7 KB
  - FIX_TRAINING_DATA_GUIDE.md        12.7 KB
  - TRAINING_DATA_SUMMARY.md          9.4 KB
  小计：50.9 KB

脚本：
  - prepare_training_data.py          13.4 KB
  - fix_training_data.py              6.7 KB
  - prepare_data_example.py           5.4 KB
  小计：25.5 KB

总计：≈ 76.4 KB（非常轻量级）
```

---

## 🎯 核心要点

### 数据格式
- ✅ **格式**：JSONL（每行一个 JSON）
- ✅ **必填**：source（音频路径）、target（转录文本）
- ✅ **编码**：UTF-8
- ✅ **路径**：使用绝对路径

### 音频要求
- ✅ **采样率**：16000 Hz
- ✅ **声道**：单声道
- ✅ **格式**：WAV、MP3、FLAC
- ✅ **质量**：清晰，SNR > 20dB

### 文本要求
- ✅ **编码**：UTF-8
- ✅ **内容**：清晰转录，无特殊符号
- ✅ **语言**：中文、英文或混合
- ✅ **长度**：建议 5-30 字

### 数据量建议
- 🟢 **最少**：5-10 小时（微调）
- 🟡 **推荐**：50-100 小时
- 🔴 **理想**：>100 小时

---

## 📖 文档导航树

```
README_DATA_PREPARATION.md (主导航页)
├── QUICK_START_TRAINING.md (快速开始)
├── DATA_FORMAT_ANALYSIS.md (深度分析)
├── FIX_TRAINING_DATA_GUIDE.md (问题修复)
└── TRAINING_DATA_SUMMARY.md (项目总结)

脚本
├── prepare_training_data.py (CSV 转换)
├── fix_training_data.py (JSONL 修复)
└── examples/prepare_data_example.py (示例)
```

---

## 🎁 额外资源

### 官方示例
- `examples/industrial_data_pretraining/fun_asr_nano/` - 模型代码
- `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/` - 完整训练示例
- `examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py` - 数据模拟工具

### 预模型位置
```bash
$MODELSCOPE_CACHE/models/
# 默认：$HOME/.cache/modelscope/models/
```

---

## ⏱️ 时间投入估计

| 任务 | 耗时 |
|------|------|
| 阅读导航页（README） | 3 分钟 |
| 阅读快速开始指南 | 5 分钟 |
| 准备数据（CSV → JSONL） | 5-10 分钟 |
| 验证数据有效性 | 2 分钟 |
| 运行完整训练流程 | 1-10 天（取决于数据量） |
| **总计（数据准备）** | **15-20 分钟** |

---

## 🔒 数据安全建议

- ✅ 定期备份原始数据
- ✅ 使用绝对路径避免丢失
- ✅ 验证数据完整性（md5sum）
- ✅ 保存多份副本

---

## 📝 最后提醒

- ✅ 所有脚本都已测试可用
- ✅ 文档都包含详细示例
- ✅ 支持快速修复和转换
- ✅ 完全离线工作（无网络依赖）

**开始使用**：
1. 从 [README_DATA_PREPARATION.md](README_DATA_PREPARATION.md) 开始
2. 选择合适的脚本运行
3. 参考文档解决问题

---

**准备好了吗？** 🚀 [立即开始](QUICK_START_TRAINING.md)

---

**版本**：1.0 | **更新时间**：2026-01-05 10:02 UTC
