# Fun-ASR-Nano 训练快速开始指南

## 概览

本文档提供了 **Fun-ASR-Nano** 模型训练的完整快速开始指南，包括数据准备、模型配置、训练运行和推理评估。

---

## 第一步：准备训练数据

### 1.1 数据格式

训练和验证数据采用 **JSONL** 格式（JSON Lines，每行一个 JSON 对象）。

**最小格式：**
```jsonl
{"source": "/path/to/audio.wav", "target": "转录文本"}
```

**标准格式（推荐）：**
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

### 1.2 音频格式要求

- **采样率**：16000 Hz（16kHz）
- **声道**：单声道（立体声自动转换）
- **格式**：WAV、MP3、FLAC 等
- **时长**：建议 5-30 秒/样本
- **质量**：清晰无损，信噪比 > 20dB（微调）

### 1.3 文本格式要求

- **编码**：UTF-8
- **语言**：中文、英文或混合
- **内容**：
  - ✓ 普通话、粤语等方言
  - ✓ 外文单词（如"machine learning"）
  - ✓ 数字、符号
  - ✗ 特殊符号（"[笑]"、"[停顿]"等）— 建议移除

### 1.4 准备 JSONL 数据

**方案 A：从 CSV 转换**
```bash
python prepare_training_data.py \
    --input data.csv \
    --output train.jsonl \
    --audio_dir /path/to/audio \
    --validate
```

**CSV 格式：**
```
source,target,speaker_id
audio/001.wav,转录内容,speaker_1
audio/002.wav,第二段转录,speaker_2
```

**方案 B：修复现有 JSONL**
```bash
# 如果 JSONL 文件有路径或格式问题，使用修复脚本
python fix_training_data.py \
    broken_train.jsonl \
    train.jsonl \
    --audio_dir /path/to/audio
```

**方案 C：手动创建**
```python
import json

data = [
    {
        "key": "train_000001",
        "source": "/data/audio/001.wav",
        "target": "这是第一条转录"
    },
    {
        "key": "train_000002",
        "source": "/data/audio/002.wav",
        "target": "这是第二条转录"
    },
]

with open('train.jsonl', 'w', encoding='utf-8') as f:
    for record in data:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
```

### 1.5 验证数据

```bash
# 检查 JSONL 有效性
python -c "
import json
import os
valid = 0
missing = 0
with open('train.jsonl') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            valid += 1
            if not os.path.isfile(record['source']):
                missing += 1
print(f'Valid: {valid}, Missing files: {missing}')
"
```

---

## 第二步：准备预模型

### 2.1 设置 ModelScope 缓存

```bash
# 设置缓存目录（可选）
export MODELSCOPE_CACHE=$HOME/.cache/modelscope

# 或指定其他路径
export MODELSCOPE_CACHE=/mnt/cache/modelscope
```

### 2.2 模型组件

Fun-ASR-Nano 由三个主要组件组成：

| 组件 | 功能 | 状态 | 说明 |
|------|------|------|------|
| **SenseVoiceEncoderSmall** | 音频编码 | 可训练/冻结 | 提取音频特征 |
| **AudioAdaptor** | 音频-文本适配 | 通常冻结 | 跨域特征适配 |
| **Qwen3-0.6B** | 文本 LLM | 通常冻结 | 生成转录文本 |

### 2.3 配置文件

选择合适的配置文件：

**8kHz 电话场景（推荐）：**
```bash
conf/config_8k_telephone.yaml
```

**16kHz 宽带场景：**
```bash
conf/config_16k_general.yaml  # 或自定义
```

---

## 第三步：数据模拟（可选，仅电话场景）

如果需要针对 8kHz 电话音频场景微调：

```bash
# 1. 模拟训练数据
python examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py \
    --input train.jsonl \
    --output train_8k.jsonl \
    --output_audio_dir ./audio_8k \
    --target_fs 8000 \
    --output_fs 16000 \
    --num_workers 64

# 2. 模拟验证数据
python examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py \
    --input val.jsonl \
    --output val_8k.jsonl \
    --output_audio_dir ./audio_8k \
    --num_workers 64
```

**数据模拟流程：**
- 16kHz 原始音频 → 8kHz 下采样 → 带通滤波(300-3400Hz) 
- G.711 编解码 → 噪声添加(SNR 15-25dB) → 16kHz 上采样

---

## 第四步：启动训练

### 4.1 8kHz 电话场景训练

```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# 运行训练脚本
bash finetune_8k_telephone.sh

# 或指定具体 stage
bash finetune_8k_telephone.sh 0 1  # stage 0-1
```

**脚本参数配置（在脚本中修改）：**

```bash
# GPU 配置
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 数据配置
source_train_data="/path/to/train.jsonl"
source_val_data="/path/to/val.jsonl"
train_data="./data_8k/train_16k.jsonl"
val_data="./data_8k/val_16k.jsonl"

# 输出目录
output_root="/output/funasr"

# 超参数
batch_size=4096
num_workers=32
max_epoch=20
learning_rate=0.00005
```

### 4.2 自定义训练脚本

创建 `train.py`：

```python
#!/usr/bin/env python3
import sys
import argparse
from funasr.tasks.asr import ASRTask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4096)
    args = parser.parse_args()
    
    # 创建 ASR 任务
    task = ASRTask.from_config(
        config_file=args.config,
        train_data=args.train_data,
        val_data=args.val_data,
        output_dir=args.output_dir,
    )
    
    # 启动训练
    task.train()

if __name__ == "__main__":
    main()
```

**运行：**
```bash
python train.py \
    --config conf/config_8k_telephone.yaml \
    --train_data train_8k.jsonl \
    --val_data val_8k.jsonl \
    --output_dir ./exp_output
```

### 4.3 训练监控

**查看训练日志：**
```bash
tail -f exp_output/train.log

# 或使用 tensorboard
tensorboard --logdir exp_output
```

**主要指标：**
- **Loss**：训练损失（应持续下降）
- **Accuracy**：文本准确率（应持续上升）
- **CER**：字错误率（应持续下降）

---

## 第五步：推理与评估

### 5.1 使用推理脚本

```bash
python inference_8k.py \
    --exp_dir exp_output \
    --data_path test.jsonl \
    --batch_size 1 \
    --gpu_id 0
```

### 5.2 单个音频推理

```python
from funasr import AutoModel

# 加载模型
model = AutoModel(model="path/to/exp_output", trust_remote_code=True)

# 推理
result = model.inference(audio_path="/path/to/audio.wav")
print(result)
```

### 5.3 评估指标

```bash
# 计算 CER、WER 等指标
python compute_metrics.py \
    --reference val.jsonl \
    --prediction pred.jsonl
```

---

## 数据规模参考

| 数据量 | 训练策略 | 学习率 | Epoch | 批大小 | 预计耗时 |
|------|----------|--------|--------|--------|-----------|
| 5-10h | 全量微调 | 5e-5 | 20-30 | 2000 | 1-2 天 |
| 10-50h | 全量微调 | 5e-5 | 15-20 | 4096 | 2-5 天 |
| 50-100h | 全量微调 | 5e-5 | 10-15 | 6000 | 5-10 天 |
| >100h | 全量微调 | 5e-5 | 5-10 | 6000 | 10+ 天 |
| <5h | LoRA 微调 | 5e-4 | 50+ | 1024 | 2-4 小时 |

---

## 常见问题解决

### Q1: 训练时 OOM（内存不足）

**解决方案：**
```yaml
# 减少批大小
batch_size: 2048  # 原 4096

# 启用梯度检查点
train_conf:
    activation_checkpoint: true

# 启用 DeepSpeed
use_deepspeed: true
```

### Q2: 训练效果不理想

**检查清单：**
1. ✓ 数据质量：检查音频和转录文本
2. ✓ 数据量：确保有足够样本（最少 5-10h）
3. ✓ 学习率：太大会不稳定，太小会收敛慢
4. ✓ 预模型：确保使用了合适的预训练模型
5. ✓ 训练步数：可能需要更多 epoch

### Q3: 推理速度慢

**优化方案：**
```bash
# 使用 FP16 加速
export TORCH_CUDA_DEVICE_ORDER=PCI_BUS_ID
python inference.py --fp16

# 使用 ONNX 导出（可选）
python export_onnx.py --model_dir exp_output
```

---

## 完整工作流示例

```bash
# 1. 准备数据
mkdir -p data
# 将 train.csv 放在 data/ 目录

# 2. 转换为 JSONL
python prepare_training_data.py \
    --input data/train.csv \
    --output data/train.jsonl \
    --audio_dir /path/to/audio

# 3. 分割训练/验证集
python -c "
import json
lines = open('data/train.jsonl').readlines()
split = int(len(lines) * 0.9)
with open('data/train_split.jsonl', 'w') as f:
    f.writelines(lines[:split])
with open('data/val_split.jsonl', 'w') as f:
    f.writelines(lines[split:])
"

# 4. 数据模拟（可选）
cd examples/industrial_data_pretraining/fun_asr_nano_8k_telephone
python data/data_simulation.py \
    --input ../../data/train_split.jsonl \
    --output ../../data/train_8k.jsonl \
    --output_audio_dir ../../data/audio_8k

# 5. 启动训练
bash finetune_8k_telephone.sh

# 6. 推理评估
python inference_8k.py \
    --exp_dir outputs/exp */
    --data_path ../../data/val_8k.jsonl
```

---

## 资源链接

- **数据格式详解**：[DATA_FORMAT_ANALYSIS.md](DATA_FORMAT_ANALYSIS.md)
- **修复指南**：[FIX_TRAINING_DATA_GUIDE.md](FIX_TRAINING_DATA_GUIDE.md)
- **官方示例**：[examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/](examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/)
- **模型代码**：[examples/industrial_data_pretraining/fun_asr_nano/model.py](examples/industrial_data_pretraining/fun_asr_nano/model.py)

---

## 支持的 GPU

- NVIDIA A100, A40, V100, L40S, RTX3090, RTX4090
- 推荐：≥ 40GB 显存（多 GPU 分布式训练）

---

## 获取帮助

遇到问题时，检查以下内容：

1. ✓ 查看训练日志：`exp_output/train.log`
2. ✓ 验证数据：`python fix_training_data.py`
3. ✓ 检查配置：`conf/config_8k_telephone.yaml`
4. ✓ 查看错误信息并搜索相关问题

---

## 总结

完整训练流程：
```
数据准备 → 格式转换 → 数据模拟 → 启动训练 → 推理评估
```

通过遵循本指南，您可以快速启动 Fun-ASR-Nano 模型的微调训练。
