# Fun-ASR-Nano 训练数据格式分析与准备指南

## 概述

本文档反推了 **Fun-ASR-Nano** 模型所需的训练数据格式，基于模型代码、示例配置和参考实现。

## 一、数据格式规范

### 1.1 JSONL 格式（主要数据格式）

训练和验证数据采用 **JSONL**（JSON Lines）格式，每行一个 JSON 对象。

#### 基础格式

```jsonl
{"source": "/path/to/audio.wav", "target": "转录文本"}
{"source": "/path/to/audio2.wav", "target": "转录文本 2"}
```

#### 完整格式（推荐）

```jsonl
{
  "key": "unique_id_001",
  "source": "/path/to/audio.wav",
  "source_len": 90,
  "target": "这是语音转录文本",
  "target_len": 8,
  "domain": "telephone",
  "speaker_id": "speaker_001"
}
```

### 1.2 JSONL 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `source` | string | ✓ | 音频文件路径（本地或 URL） |
| `target` | string | ✓ | 音频对应的转录文本 |
| `key` | string | × | 样本唯一标识符 |
| `source_len` | int | × | 音频长度（单位：秒或帧） |
| `target_len` | int | × | 转录文本长度（单位：字符或词） |
| `domain` | string | × | 数据领域标记（如"电话"、"会议"） |
| `speaker_id` | string | × | 说话人标识 |

### 1.3 音频格式要求

**原始音频格式（16kHz）：**
- **格式**：WAV、MP3、FLAC 等（支持 librosa）
- **采样率**：16000 Hz（标准宽带）
- **声道**：单声道（立体声会自动转换为单声道）
- **编码**：PCM 或 MP3

**处理流程：**
1. 16kHz 高质量音频（原始数据）
2. ↓ 数据模拟（可选）
3. 8kHz 电话信道模拟音频
4. ↓ 自动上采样
5. 16kHz 电话特性音频（用于训练）

### 1.4 文本格式要求

- **编码**：UTF-8
- **语言**：中文、英文或混合（Fun-ASR-Nano 支持多语言）
- **内容**：
  - ✓ 普通话、粤语等方言
  - ✓ 外文单词（如"machine learning"）
  - ✓ 数字、符号（自动规范化）
  - ✗ 特殊符号（如"[笑]"、"[停顿]"）- 建议移除

**示例：**
```jsonl
{"source": "/audio/sample1.wav", "target": "我是一名工程师"}
{"source": "/audio/sample2.wav", "target": "machine learning 深度学习"}
{"source": "/audio/sample3.wav", "target": "2025年1月4日"}
```

## 二、数据模拟工具

### 2.1 电话信道模拟（8kHz 场景）

使用 `data_simulation.py` 工具，从 16kHz 高质量音频生成 8kHz 电话信道模拟数据：

**模拟流程：**
1. **下采样**：16kHz → 8kHz
2. **带通滤波**：应用 300-3400 Hz 带通滤波器（电话线路频率响应）
3. **编解码压扩**：G.711 μ-law 或 A-law 压缩（模拟电话编码）
4. **噪声添加**：
   - 白噪声（SNR: 15-25 dB）
   - 工频干扰（50/60 Hz 及倍频）
5. **上采样**：8kHz → 16kHz（保持电话特性，兼容 WavFrontend）

### 2.2 数据模拟命令

```bash
python data/data_simulation.py \
    --input /path/to/train.jsonl \
    --output /output/train_8k.jsonl \
    --output_audio_dir /output/audio_8k \
    --audio_key source \
    --target_fs 8000 \
    --output_fs 16000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 15 \
    --snr_db_max 25 \
    --power_line_freq 50 \
    --num_workers 64
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 JSONL 文件 | 必填 |
| `--output` | 输出 JSONL 文件 | 必填 |
| `--output_audio_dir` | 输出音频目录 | 必填 |
| `--audio_key` | JSONL 中音频路径的键名 | `source` |
| `--target_fs` | 电话信道采样率 | `8000` |
| `--output_fs` | 最终输出采样率 | `16000` |
| `--low_freq` | 带通滤波器低频 | `300` Hz |
| `--high_freq` | 带通滤波器高频 | `3400` Hz |
| `--codec_type` | 编解码类型 | `mu-law` |
| `--snr_db_min` | 最小信噪比 | `15` dB |
| `--snr_db_max` | 最大信噪比 | `25` dB |
| `--power_line_freq` | 工频干扰 | `50` Hz（中国）或 `60` Hz（美国） |
| `--num_workers` | 并行工作线程数 | `64` |

## 三、模型数据流

### 3.1 训练时数据流

```
JSONL 文件（train.jsonl, val.jsonl）
    ↓
FunASR 数据集加载器 (IndexDataset)
    ↓
data_load_speech() 处理 [model.py:284-433]
    ├─ 数据模板转换（system, user, assistant 多轮对话）
    ├─ 音频加载 + fbank 特征提取 [model.py:354-368]
    ├─ 内容对齐处理 [model.py:333-401]
    └─ 输出张量格式化 [model.py:402-433]
    ↓
模型前向传播 (forward) [model.py:134-241]
    ├─ 音频编码：SenseVoiceEncoderSmall [model.py:248-257]
    ├─ 音频适配：AudioAdaptor [model.py:164-165]
    ├─ LLM embedding + audio embedding 融合 [model.py:149-199]
    ├─ LLM forward：Qwen3-0.6B [model.py:206-216]
    └─ Loss 计算 + Accuracy 计算 [model.py:217-241]
    ↓
优化器更新 + 日志记录
```

### 3.2 推理时数据流

```
输入数据（字符串或张量）
    ↓
构建对话上下文 [model.py:531-556]
    ├─ system 提示词："You are a helpful assistant."
    ├─ user 消息：包含语音占位符 "<|startofspeech|>!path<|endofspeech|>"
    └─ assistant 标记：初始值 "null"
    ↓
inference_prepare() [model.py:435-520]
    ├─ data_template() [model.py:259-282]
    ├─ data_load_speech() [model.py:284-433]
    ├─ 音频编码和特征提取
    └─ embedding 融合
    ↓
LLM 生成 [model.py:597-627]
    ├─ 自回归生成：generate()
    └─ 或教师强制：labels 输入
    ↓
输出：转录文本
```

## 四、数据模板格式

### 4.1 对话格式（data_template）

模型支持**多轮对话**格式，每轮包含：
- **system**：系统提示
- **user**：用户提问（可包含音频）
- **assistant**：助手回复（转录文本）

#### 对话示例

```python
data = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "转录以下语音：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>",
        "audio": "/path/to/audio.wav"  # 或 torch.Tensor
    },
    {
        "role": "assistant",
        "content": "这是转录结果"
    }
]
```

### 4.2 语音占位符说明

在 `<|startofspeech|>...<|endofspeech|>` 之间：
- **`!`** ：指示单个音频文件路径
  - `<|startofspeech|>!/path/to/audio.wav<|endofspeech|>`
- **`!!`** ：指示 torch.Tensor 音频数据
  - `<|startofspeech|>!!<|endofspeech|>`（需要通过 `"audio"` 字段传入）
- **空值** ：不处理音频（仅进行文本转录）

## 五、特征提取配置

### 5.1 WavFrontend 配置

```yaml
frontend: WavFrontend
frontend_conf:
    fs: 16000              # 采样率：16kHz
    window: hamming        # 窗函数：汉明窗
    n_mels: 80            # Mel 频谱频道数：80
    frame_length: 25      # 帧长：25 ms
    frame_shift: 10       # 帧移：10 ms
    lfr_m: 7              # LFR M 参数
    lfr_n: 6              # LFR N 参数（低帧率特征提取）
```

### 5.2 特征处理流程

```
原始音频（16kHz）
    ↓
预处理：DC 移除、预加重
    ↓
分帧：25ms 帧长，10ms 帧移
    ↓
Mel 特征提取：80 channel
    ↓
低帧率特征（LFR）：帧合并
    ↓
SpecAug：时间扭曲、频域掩码、时域掩码
    ↓
模型输入：[B, T, 80]
```

## 六、数据集配置

### 6.1 FunASR 数据集参数

```yaml
dataset: FunASR
dataset_conf:
    index_ds: FunASR                           # 索引方式
    batch_sampler: BatchSampler                # 批采样器
    batch_type: token                          # 批类型：token-based
    batch_size: 6000                           # 批大小：6000 tokens
    max_token_length: 1024                     # 最大 token 长度
    shuffle: true                              # 是否打乱
    sort_size: 1024                            # 排序缓冲区大小
    batch_size_scale_ratio_max: 2              # 批大小缩放比例
    num_workers: 4                             # 数据加载工作线程
    audio_adaptor_downsample_rate: 1           # 音频适配器下采样率
    audio_encoder_downsample_rate: 2           # 编码器下采样率
    data_split_num: 512                        # 数据分割数
    batch_size_sample_max: 15                  # 最大样本批大小
    retry: 20                                  # 加载重试次数
```

### 6.2 Token-based Batching

- **目标**：每批约 6000 个 token（文本 + 音频特征）
- **优势**：
  - 充分利用 GPU 内存
  - 自动处理不同长度的音频和文本
  - 提高训练效率

### 6.3 Tokenizer 配置

```yaml
tokenizer: HuggingfaceTokenizer
tokenizer_conf:
    unk_symbol: <unk>
    init_param_path: Qwen3-0.6B  # Qwen 模型的分词器
```

## 七、模型架构与数据要求

### 7.1 模型组件

| 组件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **SenseVoiceEncoderSmall** | 音频编码 | [B, T, 80] | [B, T', 512] |
| **AudioAdaptor** | 跨域适配 | [B, T', 512] | [B, T', 1024] |
| **Qwen3-0.6B** | 文本 LLM | [B, L, 1024] | logits |
| **LLM Embedding** | 文本嵌入 | input_ids | [B, L, 1024] |

### 7.2 数据融合机制

```
音频编码：[B, T', 512] + AudioAdaptor ⟹ [B, T'_adapted, 1024]
                                            ↓
                                        插入到 input_ids
                                        对应位置
                                            ↓
文本嵌入：[B, L, 1024] + 音频特征 ⟹ 融合 embedding
                                            ↓
                                        LLM forward
```

**关键点：**
- 音频特征通过 **fbank_beg** 和 **fake_token_len** 定位
- 每个音频对应的 token 数由模型自动计算
- 支持**多轮对话**中多个音频

## 八、训练配置示例

### 8.1 微调训练参数

```yaml
train_conf:
    accum_grad: 1              # 梯度累积步数
    grad_clip: 5               # 梯度裁剪
    max_epoch: 20              # 最大 epoch
    keep_nbest_models: 10      # 保留最佳模型数

optim: adamw                   # 优化器
optim_conf:
    lr: 0.00005                # 学习率：5e-5（微调）
    weight_decay: 0.01         # 权重衰减
    betas: [0.9, 0.999]

scheduler: warmuplr            # 学习率调度器
scheduler_conf:
    warmup_steps: 1000         # 预热步数
```

### 8.2 推荐参数设置

**大规模数据（>100h）：**
- learning_rate: 5e-5
- batch_size: 4096-6000 tokens
- max_epoch: 10-15
- warmup_steps: 1000-2000

**小规模数据（10-100h）：**
- learning_rate: 1e-4-5e-5
- batch_size: 2000-4000 tokens
- max_epoch: 20-30
- warmup_steps: 500-1000

**极小规模数据（<10h）：**
- learning_rate: 5e-4
- batch_size: 1000-2000 tokens
- max_epoch: 50-100
- warmup_steps: 100-500
- 可考虑 LoRA 微调

## 九、预模型与缓存

### 9.1 ModelScope 缓存配置

```bash
# 设置缓存目录
export MODELSCOPE_CACHE=/path/to/cache
export MODELSCOPE_CACHE=$HOME/.cache/modelscope

# 默认位置：$HOME/.cache/modelscope/models
```

### 9.2 预加载模型

**8K 电话场景推荐配置：**

```yaml
# 音频编码器
audio_encoder: SenseVoiceEncoderSmall
audio_encoder_conf:
    hub: ms                    # ModelScope hub
    freeze: false              # 非冻结（可继续训练）
    output_size: 512

# LLM
llm: Qwen3-0.6b
llm_conf:
    hub: hf                    # HuggingFace hub
    init_param_path: Qwen3-0.6B
    freeze: true               # 冻结（保持预训练知识）

# 音频适配器
audio_adaptor: Transformer
audio_adaptor_conf:
    freeze: true               # 冻结（保持通用适配能力）
    encoder_dim: 512
    llm_dim: 1024
    n_layer: 2
```

### 9.3 模型下载示例

```python
from funasr import AutoModel

# 自动下载到 $MODELSCOPE_CACHE
model = AutoModel(model="FunAudioLLM/Fun-ASR-Nano-2512", model_revision="master")
```

## 十、数据质量检查清单

### 10.1 预处理检查

- [ ] 所有 JSONL 行都是有效的 JSON
- [ ] `source` 字段指向存在的文件或有效 URL
- [ ] `target` 字段非空且不含特殊符号
- [ ] 音频文件都可以被 librosa 读取
- [ ] 音频采样率为 16000 Hz 或能被自动转换

### 10.2 数据分布检查

- [ ] 训练集和验证集没有重叠
- [ ] 验证集规模合理（通常为训练集的 5-10%）
- [ ] 文本长度分布合理（避免过长）
- [ ] 音频长度分布合理（避免过长）
- [ ] 说话人分布均衡（如有多说话人）

### 10.3 域特异性检查

- [ ] 电话场景：音频包含噪声、回声、混响
- [ ] 特殊术语：关键词是否覆盖
- [ ] 口语特征：叠词、缩写、非标准表达
- [ ] 多语言混合：外文比例合理

## 十一、数据准备步骤总结

### 标准流程

```bash
# 1. 准备 16kHz 原始数据
# 创建 train.jsonl 和 val.jsonl

# 2. 生成 8kHz 电话模拟数据（可选，仅需要电话场景）
python data/data_simulation.py \
    --input /path/to/train.jsonl \
    --output /output/train_8k.jsonl \
    --output_audio_dir /output/audio_8k \
    --num_workers 64

python data/data_simulation.py \
    --input /path/to/val.jsonl \
    --output /output/val_8k.jsonl \
    --output_audio_dir /output/audio_8k \
    --num_workers 64

# 3. 启动训练
bash finetune_8k_telephone.sh

# 4. 评估和推理
python inference_8k.py \
    --exp_dir /path/to/exp \
    --data_path /path/to/test.jsonl
```

## 十二、常见问题

### Q1: JSONL 文件路径应该是绝对路径还是相对路径？
**A**: 建议使用绝对路径，避免加载错误。如果使用相对路径，确保在脚本执行时工作目录正确。

### Q2: 音频文件可以用 URL 吗？
**A**: 可以，但需要网络连接。建议用于公开数据集，本地数据使用本地路径。

### Q3: 8kHz 电话模拟和 16kHz 直接训练哪个效果好？
**A**: 取决于目标场景：
- **8kHz 电话数据**：在电话场景准确率高
- **16kHz 宽带数据**：在多种场景泛化性好
- **混合**：最佳方案，同时保证两个场景的准确率

### Q4: 最小数据量是多少？
**A**: 
- 推荐最少 **10 小时**（微调）
- 最小可接受 **5 小时**（带 LoRA）
- **1-2 小时**：只能进行特定场景过拟合

### Q5: 如何处理多说话人数据？
**A**: 
1. 在 JSONL 中添加 `speaker_id` 字段（可选）
2. 模型会自动学习说话人变化
3. 不需要特殊处理

## 参考

- 模型代码：`/workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano/model.py`
- 电话训练示例：`/workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/`
- 数据模拟工具：`data_simulation.py`
- ModelScope 参考：https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano
