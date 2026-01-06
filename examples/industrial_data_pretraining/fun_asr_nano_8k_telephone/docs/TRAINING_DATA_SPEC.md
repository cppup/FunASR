# Fun-ASR-Nano 8kHz 电话场景微调 - 训练数据格式规范

## 一、数据格式概述

Fun-ASR-Nano 使用 **JSONL（JSON Lines）** 格式的训练数据，每行一个 JSON 对象。

### 核心数据结构

```json
{
  "key": "sample_001",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},
    {"role": "assistant", "content": "这是语音转写的文本内容"}
  ],
  "speech_length": 1200,
  "text_length": 50
}
```

### 字段说明

| 字段 | 类型 | 是否必须 | 说明 |
|------|------|----------|------|
| `key` | string | 可选 | 样本唯一标识符 |
| `messages` | array | **必须** | 对话消息列表 |
| `speech_length` | int | **必须** | 音频帧数（fbank 帧数，需 ≥10，≤8000） |
| `text_length` | int | **必须** | 文本字符数（需 ≥1，≤2048） |
| `hotwords` | string | 可选 | 热词列表，空格分隔 |
| `asr_hotwords` | string | 可选 | ASR 热词 |
| `one_pass_result` | string | 可选 | 一阶段识别结果（用于 2-pass） |
| `noised` | bool | 可选 | 是否已添加噪声（避免重复加噪） |

---

## 二、messages 消息格式

### 2.1 基础 ASR 任务（单轮）

```json
{
  "key": "asr_sample_001",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},
    {"role": "assistant", "content": "今天天气很好，适合出门散步。"}
  ],
  "speech_length": 800,
  "text_length": 15
}
```

### 2.2 多轮对话（Multi-turn）

```json
{
  "key": "multiturn_001",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio1.wav<|endofspeech|>"},
    {"role": "assistant", "content": "第一段语音的转写内容"},
    {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio2.wav<|endofspeech|>"},
    {"role": "assistant", "content": "第二段语音的转写内容"}
  ],
  "speech_length": [600, 500],
  "text_length": [20, 18]
}
```

### 2.3 带热词的 ASR

```json
{
  "key": "hotword_sample",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},
    {"role": "assistant", "content": "请拨打客服热线咨询订单问题。"}
  ],
  "speech_length": 750,
  "text_length": 16,
  "hotwords": "客服热线 订单 咨询",
  "asr_hotwords": "客服热线 订单号"
}
```

---

## 三、音频路径格式

音频路径使用特殊标记包裹：

```
<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
```

**关键点：**
- `<|startofspeech|>` 和 `<|endofspeech|>` 是音频边界标记
- `!` 表示后面是音频文件路径
- 支持的格式：`.wav`, `.mp3`, `.flac`, `.m4a` 等（通过 soundfile 加载）

---

## 四、User Prompt 任务模板

Fun-ASR-Nano 支持多种任务提示词：

### 4.1 标准中文转写
```
语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
```

### 4.2 标准英文转写
```
Speech transcription:<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
```

### 4.3 指定语言转写
```
语音转写成中文：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
Transcribe speech into Chinese:<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
语音转写成英文：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
Transcribe speech into English:<|startofspeech|>!/path/to/audio.wav<|endofspeech|>
```

---

## 五、8kHz 电话场景数据准备

### 5.1 方案一：从 16kHz 数据模拟

使用 `data/data_simulation.py` 工具：

```bash
python data/data_simulation.py \
    --input /path/to/16khz/train.jsonl \
    --output /path/to/8khz/train.jsonl \
    --output_audio_dir /path/to/8khz/audio \
    --audio_key source \
    --target_fs 8000 \
    --output_fs 16000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 15 \
    --snr_db_max 25
```

**输入格式（简化版）：**
```json
{"source": "/path/to/16khz/audio.wav", "target": "转写文本"}
```

**输出格式（完整版）：**
```json
{"source": "/path/to/8khz/audio_8k.wav", "target": "转写文本"}
```

### 5.2 方案二：真实 8kHz 电话数据

直接使用真实电话录音，但需要 **上采样到 16kHz** 后再用于训练。

数据准备脚本：

```python
import json
import soundfile as sf
from scipy import signal
import os

def prepare_8k_data(input_jsonl, output_jsonl, output_audio_dir):
    """将8kHz数据上采样到16kHz并生成训练JSONL"""
    os.makedirs(output_audio_dir, exist_ok=True)
    
    with open(input_jsonl, 'r') as fin, open(output_jsonl, 'w') as fout:
        for line in fin:
            data = json.loads(line.strip())
            audio_path = data['source']
            
            # 加载音频
            audio, sr = sf.read(audio_path)
            
            # 如果是8kHz，上采样到16kHz
            if sr == 8000:
                num_samples = int(len(audio) * 16000 / 8000)
                audio_16k = signal.resample(audio, num_samples)
                
                # 保存16kHz版本
                basename = os.path.basename(audio_path)
                output_path = os.path.join(output_audio_dir, basename.replace('.wav', '_16k.wav'))
                sf.write(output_path, audio_16k, 16000)
                
                data['source'] = output_path
            
            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
```

### 5.3 转换为 Fun-ASR-Nano 格式

从简化格式转换为完整 messages 格式：

```python
import json

def convert_to_funasr_format(input_jsonl, output_jsonl, frontend_frame_shift=10, frontend_lfr_n=6):
    """
    将简化格式转换为Fun-ASR-Nano训练格式
    
    Args:
        input_jsonl: 输入JSONL，格式 {"source": "audio.wav", "target": "文本"}
        output_jsonl: 输出JSONL，完整的messages格式
        frontend_frame_shift: WavFrontend帧移（默认10ms）
        frontend_lfr_n: LFR因子（默认6）
    """
    import soundfile as sf
    
    with open(input_jsonl, 'r') as fin, open(output_jsonl, 'w') as fout:
        for idx, line in enumerate(fin):
            data = json.loads(line.strip())
            audio_path = data['source']
            target_text = data['target']
            
            # 获取音频长度
            info = sf.info(audio_path)
            duration_sec = info.duration
            sample_rate = info.samplerate
            
            # 计算fbank帧数
            # 帧长25ms，帧移10ms → 每秒约100帧
            # LFR因子6 → 实际帧数 = 原始帧数 / 6
            num_frames = int(duration_sec * 1000 / frontend_frame_shift)
            speech_length = num_frames // frontend_lfr_n
            
            # 构建完整格式
            output = {
                "key": f"sample_{idx:08d}",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"语音转写：<|startofspeech|>!{audio_path}<|endofspeech|>"},
                    {"role": "assistant", "content": target_text}
                ],
                "speech_length": speech_length,
                "text_length": len(target_text)
            }
            
            # 可选：添加热词
            if 'hotwords' in data:
                output['hotwords'] = data['hotwords']
            
            fout.write(json.dumps(output, ensure_ascii=False) + '\n')
```

---

## 六、训练配置要点

### 6.1 关键配置参数

```yaml
# config_8k_telephone.yaml

# Frontend - 必须是16kHz
frontend: WavFrontend
frontend_conf:
    fs: 16000  # 关键：8kHz数据需要上采样到16kHz
    n_mels: 80
    lfr_m: 7
    lfr_n: 6

# 数据集配置
dataset: FunASR
dataset_conf:
    index_ds: FunASR
    max_source_length: 8000   # 最大音频帧数
    min_source_length: 10     # 最小音频帧数
    max_target_length: 2048   # 最大文本长度
    batch_size: 4096          # token-based batching
    batch_type: token
```

### 6.2 数据过滤条件

训练时会自动过滤以下数据：
- `speech_length < 10` 或 `speech_length > 8000`
- `text_length < 1` 或 `text_length > 2048`
- 加载音频失败的样本
- 超过 `max_token_length` 的样本

---

## 七、完整数据处理流程

```
┌──────────────────────────────────────────────────────────┐
│                     原始数据准备                          │
├──────────────────────────────────────────────────────────┤
│  方案A：16kHz高质量数据                                   │
│  ├── 使用 data_simulation.py 模拟电话信道                │
│  └── 输出：保留电话特征的16kHz音频                       │
│                                                          │
│  方案B：真实8kHz电话录音                                  │
│  ├── 上采样到16kHz（保留电话特征）                       │
│  └── 输出：16kHz音频                                     │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                   格式转换                                │
├──────────────────────────────────────────────────────────┤
│  输入：{"source": "audio.wav", "target": "文本"}          │
│                           ↓                              │
│  输出：完整 messages 格式                                │
│  {                                                       │
│    "key": "sample_001",                                  │
│    "messages": [...],                                    │
│    "speech_length": 800,                                 │
│    "text_length": 20                                     │
│  }                                                       │
└──────────────────────────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────┐
│                   训练脚本                                │
├──────────────────────────────────────────────────────────┤
│  bash finetune_8k_telephone.sh 1 1                       │
│                                                          │
│  关键配置：                                              │
│  - frontend_conf.fs: 16000 (必须)                        │
│  - audio_encoder_conf.freeze: false                      │
│  - llm_conf.freeze: true + use_lora: true               │
│  - audio_adaptor_conf.freeze: false                      │
│  - ctc_decoder_conf.freeze: false                        │
└──────────────────────────────────────────────────────────┘
```

---

## 八、示例数据

### 8.1 电话客服场景

```json
{"key":"cs_001","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"语音转写：<|startofspeech|>!/data/8k_telephone/audio/cs_001_16k.wav<|endofspeech|>"},{"role":"assistant","content":"您好，请问有什么可以帮您的？"}],"speech_length":450,"text_length":12,"hotwords":"客服 订单 退款"}
{"key":"cs_002","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"语音转写：<|startofspeech|>!/data/8k_telephone/audio/cs_002_16k.wav<|endofspeech|>"},{"role":"assistant","content":"我想查询一下我的订单状态。"}],"speech_length":520,"text_length":13,"hotwords":"订单状态 查询"}
{"key":"cs_003","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"语音转写：<|startofspeech|>!/data/8k_telephone/audio/cs_003_16k.wav<|endofspeech|>"},{"role":"assistant","content":"请您提供一下订单号，我帮您查询。"}],"speech_length":680,"text_length":16,"hotwords":"订单号 查询"}
```

### 8.2 电话销售场景

```json
{"key":"sales_001","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"语音转写：<|startofspeech|>!/data/8k_telephone/audio/sales_001_16k.wav<|endofspeech|>"},{"role":"assistant","content":"您好，我是某某公司的销售代表。"}],"speech_length":550,"text_length":15}
{"key":"sales_002","messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"语音转写：<|startofspeech|>!/data/8k_telephone/audio/sales_002_16k.wav<|endofspeech|>"},{"role":"assistant","content":"我们公司最近推出了一款新产品，想给您介绍一下。"}],"speech_length":780,"text_length":23}
```

---

## 九、数据验证脚本

```python
#!/usr/bin/env python3
"""验证训练数据格式是否正确"""

import json
import os
import sys

def validate_jsonl(jsonl_path, check_audio=False):
    """验证JSONL数据格式"""
    errors = []
    warnings = []
    valid_count = 0
    
    with open(jsonl_path, 'r') as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: JSON解析错误 - {e}")
                continue
            
            # 检查必须字段
            if 'messages' not in data:
                errors.append(f"Line {line_no}: 缺少 'messages' 字段")
                continue
            
            if 'speech_length' not in data:
                warnings.append(f"Line {line_no}: 缺少 'speech_length' 字段")
            
            if 'text_length' not in data:
                warnings.append(f"Line {line_no}: 缺少 'text_length' 字段")
            
            # 检查 messages 结构
            messages = data['messages']
            has_user = False
            has_assistant = False
            
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    errors.append(f"Line {line_no}: message 缺少 'role' 或 'content'")
                    continue
                
                if msg['role'] == 'user':
                    has_user = True
                    content = msg['content']
                    if '<|startofspeech|>' in content:
                        # 检查音频路径格式
                        if '!' not in content:
                            errors.append(f"Line {line_no}: 音频路径格式错误，缺少 '!'")
                        elif check_audio:
                            # 提取音频路径并检查
                            import re
                            match = re.search(r'<\|startofspeech\|>!(.+?)<\|endofspeech\|>', content)
                            if match:
                                audio_path = match.group(1)
                                if not os.path.exists(audio_path):
                                    errors.append(f"Line {line_no}: 音频文件不存在 - {audio_path}")
                
                if msg['role'] == 'assistant':
                    has_assistant = True
            
            if not has_user:
                errors.append(f"Line {line_no}: 缺少 'user' 角色的消息")
            if not has_assistant:
                errors.append(f"Line {line_no}: 缺少 'assistant' 角色的消息")
            
            if not errors or errors[-1].split(':')[0] != f"Line {line_no}":
                valid_count += 1
    
    # 输出报告
    print(f"\n{'='*60}")
    print(f"数据验证报告: {jsonl_path}")
    print(f"{'='*60}")
    print(f"有效样本: {valid_count}")
    print(f"错误数量: {len(errors)}")
    print(f"警告数量: {len(warnings)}")
    
    if errors:
        print(f"\n错误列表 (前10条):")
        for err in errors[:10]:
            print(f"  ❌ {err}")
    
    if warnings:
        print(f"\n警告列表 (前10条):")
        for warn in warnings[:10]:
            print(f"  ⚠️ {warn}")
    
    print(f"{'='*60}\n")
    
    return len(errors) == 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_data.py <jsonl_path> [--check-audio]")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    check_audio = '--check-audio' in sys.argv
    
    success = validate_jsonl(jsonl_path, check_audio)
    sys.exit(0 if success else 1)
```

---

## 十、常见问题

### Q1: speech_length 如何计算？

```python
# 假设音频采样率为16kHz
# WavFrontend配置：frame_shift=10ms, lfr_n=6

duration_sec = audio_samples / sample_rate  # 音频时长（秒）
num_frames = int(duration_sec * 1000 / 10)  # 原始帧数（每10ms一帧）
speech_length = num_frames // 6              # LFR后的帧数
```

### Q2: 为什么 8kHz 数据需要上采样到 16kHz？

Fun-ASR-Nano 的 WavFrontend 预训练在 16kHz 数据上，前端的 FFT 参数（n_fft=400, hop_length=160）假设 16kHz 输入。8kHz 音频上采样后保留了电话信道特征（频谱截断、编解码失真），但与前端兼容。

### Q3: 可以直接用 8kHz 前端吗？

不推荐。修改前端采样率会导致与预训练编码器不兼容，需要大量数据重新训练。上采样方案可以复用预训练知识，训练效率更高。

---

## 版本历史

- **2025-01-05**: 初始版本，基于 Fun-ASR-Nano-2512 模型
