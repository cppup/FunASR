# Fun-ASR-Nano 训练数据反推总结

## 核心发现

通过对 `model.py` 代码、配置文件和示例的详细分析，我已经完整反推了 **Fun-ASR-Nano** 所需的训练数据格式。

### 主要要点

| 方面 | 要求 | 说明 |
|------|------|------|
| **数据格式** | JSONL（JSON Lines） | 每行一个 JSON 对象 |
| **必填字段** | `source`, `target` | 音频路径、转录文本 |
| **音频格式** | 16kHz, mono, WAV/MP3 | 自动重采样至 16kHz |
| **文本格式** | UTF-8, 中文/英文 | 支持多语言混合 |
| **典型数据量** | 10-100h | 微调所需规模 |
| **预模型位置** | $MODELSCOPE_CACHE/models | 自动下载缓存 |

---

## 已生成的资源

### 1. 数据格式分析文档
📄 **DATA_FORMAT_ANALYSIS.md** (10.7KB)

内容：
- 数据格式规范与示例
- 模型数据流完整说明
- 特征提取配置详解
- 预模型与缓存配置
- 质量检查清单

### 2. 修复指南
📄 **FIX_TRAINING_DATA_GUIDE.md** (12.7KB)

内容：
- 问题诊断方法
- 四种修复方案（CSV转换、现有JSONL修复、手动创建、示例脚本）
- 完整验证检查清单
- 常见错误及解决方案
- 最佳实践建议

### 3. 快速开始指南
📄 **QUICK_START_TRAINING.md** (8.1KB)

内容：
- 完整5步训练流程
- 数据准备、预模型、数据模拟、训练、推理
- 数据规模参考表
- 常见问题解决
- 完整工作流示例

### 4. 数据准备脚本
🐍 **prepare_training_data.py** (13.4KB)

功能：
- CSV → JSONL 格式转换
- 路径处理与验证
- 数据验证与报告

使用：
```bash
python prepare_training_data.py \
    --input data.csv \
    --output train.jsonl \
    --audio_dir /path/to/audio \
    --validate
```

### 5. 数据修复脚本
🐍 **fix_training_data.py** (6.7KB)

功能：
- 修复JSONL格式问题
- 转换相对路径为绝对路径
- 验证音频文件存在性
- 清理文本内容

使用：
```bash
python fix_training_data.py \
    broken.jsonl \
    fixed.jsonl \
    --audio_dir /path/to/audio
```

### 6. 数据准备示例
🐍 **examples/prepare_data_example.py** (5.4KB)

功能：
- 演示如何手动准备数据
- 包含详细注释和使用说明

---

## JSONL 数据格式规范

### 最小格式
```jsonl
{"source": "/path/to/audio.wav", "target": "转录文本"}
```

### 标准格式（推荐）
```jsonl
{
  "key": "sample_000001",
  "source": "/path/to/audio.wav",
  "source_len": 10,
  "target": "这是转录文本",
  "target_len": 5
}
```

### 完整格式（最佳）
```jsonl
{
  "key": "sample_000001",
  "source": "/path/to/audio.wav",
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
| source | string | ✓ | 音频文件绝对路径 |
| target | string | ✓ | UTF-8 编码的转录文本 |
| key | string | × | 样本唯一标识符 |
| source_len | int | × | 音频时长（秒） |
| target_len | int | × | 文本长度（字符） |
| domain | string | × | 数据领域标记 |
| speaker_id | string | × | 说话人标识 |

---

## 关键技术点

### 1. 数据融合机制（model.py:134-241）

```
输入数据流
    ├─ 音频部分
    │  ├─ load_audio_text_image_video() → 加载音频
    │  ├─ extract_fbank() → 提取 Mel 频谱
    │  ├─ SenseVoiceEncoderSmall → 编码 [B,T,512]
    │  └─ AudioAdaptor → 适配 [B,T',1024]
    │
    ├─ 文本部分
    │  ├─ tokenizer.encode() → token IDs
    │  └─ LLM embedding → 文本向量 [B,L,1024]
    │
    └─ 融合部分
       ├─ 通过 fbank_beg 定位音频插入位置
       ├─ 通过 fake_token_len 确定音频 token 数
       └─ 融合后送入 LLM 前向计算
```

### 2. 特征提取参数（model.py:70-77）

```yaml
frontend: WavFrontend
  fs: 16000           # 采样率
  n_mels: 80          # Mel 频道数
  frame_length: 25ms  # 帧长
  frame_shift: 10ms   # 帧移
  lfr_n: 6            # 低帧率参数
```

### 3. 对话格式支持（model.py:259-282）

支持多轮对话，每轮包含：
- system（系统提示）
- user（用户输入 + 音频）
- assistant（助手回复）

例如：
```json
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "转写语音：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},
  {"role": "assistant", "content": "转录结果"}
]
```

### 4. 数据模拟流程（data_simulation.py）

对于 8kHz 电话场景：
```
16kHz 原始 → 8kHz 下采样 → 带通滤波(300-3400Hz) 
    → G.711 编解码 → 噪声添加 → 16kHz 上采样
```

---

## 推荐使用流程

### 快速修复现有数据
```bash
# 如果已有 JSONL 但格式有问题
python fix_training_data.py \
    old_train.jsonl \
    train.jsonl \
    --audio_dir /data/audio

python fix_training_data.py \
    old_val.jsonl \
    val.jsonl \
    --audio_dir /data/audio
```

### 从 CSV 准备数据
```bash
# 如果有 CSV 文件
python prepare_training_data.py \
    --input train.csv \
    --output train.jsonl \
    --audio_dir /data/audio \
    --validate
```

### 从零开始创建
```bash
# 参考示例脚本
python examples/prepare_data_example.py
```

### 电话场景数据模拟
```bash
python examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py \
    --input train.jsonl \
    --output train_8k.jsonl \
    --output_audio_dir ./audio_8k \
    --num_workers 64
```

---

## 验证检查命令

```bash
# 1. 检查 JSONL 格式有效性
python -c "
import json
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        if line.strip():
            try:
                json.loads(line)
            except:
                print(f'Line {i}: Invalid JSON')
"

# 2. 检查必填字段
python -c "
import json
import os
for i, line in enumerate(open('train.jsonl')):
    if line.strip():
        r = json.loads(line)
        if 'source' not in r or 'target' not in r:
            print(f'Line {i}: Missing fields')
        elif not os.path.isfile(r['source']):
            print(f'Line {i}: File not found {r[\"source\"]}')
"

# 3. 显示样本
python -c "
import json
for i, line in enumerate(open('train.jsonl')):
    if i >= 3: break
    r = json.loads(line)
    print(f'{i}: {r[\"target\"][:50]}')
"

# 4. 统计信息
python -c "
import json
lens = []
for line in open('train.jsonl'):
    if line.strip():
        lens.append(len(json.loads(line)['target']))
print(f'Samples: {len(lens)}, Avg text len: {sum(lens)/len(lens):.1f}')
"
```

---

## 常见错误及解决

### 路径问题
```
❌ {"source": "audio/001.wav"}  # 相对路径
✓ {"source": "/data/audio/001.wav"}  # 绝对路径
```

### 文件存在性
```bash
# 检查并修复路径
ls -la /path/to/audio/*.wav

# 转换为绝对路径
python fix_training_data.py input.jsonl output.jsonl
```

### 文本质量
```
❌ {"target": "[笑] [停顿]"}  # 特殊符号
✓ {"target": "清晰的转录文本"}  # 干净文本
```

### 编码问题
```bash
# 确保 UTF-8 编码
file -i train.jsonl  # 应显示 charset=utf-8
```

---

## 预模型与环境

### ModelScope 缓存
```bash
# 设置缓存目录
export MODELSCOPE_CACHE=$HOME/.cache/modelscope

# 或
export MODELSCOPE_CACHE=/mnt/cache/modelscope
```

### 预加载模型
```python
from funasr import AutoModel

# 自动下载到缓存
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512")
```

### 推荐配置
```yaml
audio_encoder: SenseVoiceEncoderSmall
llm: Qwen3-0.6b
audio_adaptor: Transformer
```

---

## 训练参数参考

| 参数 | 大数据(>100h) | 中等(50h) | 小数据(10h) | 极小(<5h) |
|------|----------|---------|---------|-----------|
| 学习率 | 5e-5 | 5e-5 | 1e-4 | 5e-4 |
| Epoch | 5-10 | 10-15 | 20-30 | 50-100 |
| 批大小 | 6000 | 4096 | 2000 | 1024 |
| 训练时间 | 10+ 天 | 2-5 天 | 1-2 天 | 2-4 小时 |
| 策略 | 全量微调 | 全量微调 | 全量微调 | LoRA 微调 |

---

## 文件列表总结

```
/workspace/share/LLMFunASR/
├── DATA_FORMAT_ANALYSIS.md          # 数据格式完整分析
├── FIX_TRAINING_DATA_GUIDE.md       # 修复指南（常见问题解决）
├── QUICK_START_TRAINING.md          # 快速开始流程
├── TRAINING_DATA_SUMMARY.md         # 本文档
├── prepare_training_data.py         # CSV → JSONL 转换脚本
├── fix_training_data.py             # JSONL 修复脚本
└── examples/
    └── prepare_data_example.py      # 数据准备示例
```

---

## 后续步骤

1. **准备数据**：
   - 使用 `prepare_training_data.py` 或 `fix_training_data.py`
   - 验证 JSONL 格式有效性

2. **配置模型**：
   - 选择合适的 config YAML
   - 调整超参数（学习率、batch_size 等）

3. **数据模拟**（可选）：
   - 如需电话场景，运行 `data_simulation.py`

4. **启动训练**：
   ```bash
   bash examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/finetune_8k_telephone.sh
   ```

5. **推理评估**：
   - 运行推理脚本
   - 计算 CER/WER 等指标

---

## 联系与支持

- 📖 详细文档参考：`DATA_FORMAT_ANALYSIS.md`
- 🔧 问题排查参考：`FIX_TRAINING_DATA_GUIDE.md`
- ⚡ 快速上手参考：`QUICK_START_TRAINING.md`
- 💻 官方代码：`examples/industrial_data_pretraining/fun_asr_nano/`

---

**最后更新**：2026-01-04 14:31 UTC

所有文档和脚本已准备完毕，可以开始准备数据进行训练！
