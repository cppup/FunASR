# Fun-ASR-Nano 8kHz 电话场景微调训练方案

[English](README.md) | 简体中文

## 简介

针对 8kHz 窄带电话信道场景（如客服中心、电话销售、VoIP 通信等），本方案提供了 Fun-ASR-Nano 模型的专门微调训练工具链。相比标准 16kHz 宽带音频，8kHz 窄带音频面临以下挑战：

- **频谱截断**：奈奎斯特频率仅 4kHz，高频分量丢失
- **编解码失真**：G.711 (μ-law/A-law) 压扩带来的量化噪声
- **信道噪声**：线路底噪、电流音 (50/60Hz) 干扰
- **口语化特征**：叠词、断续、不规范表达

本方案通过数据模拟、针对性训练配置和推理工具，帮助用户快速实现 8kHz 电话场景的高精度语音识别。

## 特性

- ✅ **数据模拟工具**：从 16kHz 高质量语料生成 8kHz 电话信道模拟数据
- ✅ **训练配置优化**：针对 8kHz 窄带信号优化的模型配置
- ✅ **分布式训练**：支持多 GPU 和 DeepSpeed 加速训练
- ✅ **推理评估**：支持单文件、批量和 JSONL 评估，输出 CER/WER 指标
- ✅ **热词增强**：支持热词偏置解码，提升领域术语识别准确率

## 目录结构

```
fun_asr_nano_8k_telephone/
├── README_zh.md                    # 中文文档（本文件）
├── requirements.txt                # Python 依赖
├── conf/
│   └── config_8k_telephone.yaml    # 8kHz 专用训练配置
├── data/
│   └── data_simulation.py          # 16kHz→8kHz 数据模拟工具
├── finetune_8k_telephone.sh        # 微调训练脚本
└── inference_8k.py                 # 推理脚本
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 数据准备

#### 1.1 准备 16kHz 高质量数据

首先准备 JSONL 格式的 16kHz 高质量训练数据，格式如下：

```jsonl
{"source": "/path/to/audio1.wav", "target": "这是第一段语音的文本"}
{"source": "/path/to/audio2.wav", "target": "这是第二段语音的文本"}
```

#### 1.2 生成 8kHz 电话信道模拟数据

使用数据模拟工具将 16kHz 音频转换为 8kHz 电话信道模拟数据：

```bash
python data/data_simulation.py \
    --input /path/to/16khz/train.jsonl \
    --output /path/to/8khz/train.jsonl \
    --output_audio_dir /path/to/8khz/audio \
    --audio_key source \
    --target_fs 8000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 15 \
    --snr_db_max 25 \
    --power_line_freq 50 \
    --bg_noise_scp /path/to/noise.scp \
    --bg_noise_snr_min 5 \
    --bg_noise_snr_max 20
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入 JSONL 文件路径 | 必填 |
| `--output` | 输出 JSONL 文件路径 | 必填 |
| `--output_audio_dir` | 输出音频文件目录 | 必填（JSONL 模式） |
| `--audio_key` | JSONL 中音频路径的键名 | `source` |
| `--target_fs` | 目标采样率（Hz） | `8000` |
| `--low_freq` | 带通滤波器低频截止（Hz） | `300` |
| `--high_freq` | 带通滤波器高频截止（Hz） | `3400` |
| `--codec_type` | 编解码类型（`mu-law` 或 `a-law`） | `mu-law` |
| `--snr_db_min` | 电话线路噪声最小信噪比（dB） | `15` |
| `--snr_db_max` | 电话线路噪声最大信噪比（dB） | `25` |
| `--power_line_freq` | 工频干扰频率（50 或 60 Hz） | `50` |
| `--no_noise` | 禁用电话线路噪声添加 | `False` |
| `--no_codec` | 禁用编解码压扩 | `False` |
| `--bg_noise_scp` | 背景噪声 SCP 文件路径（格式：utt_id /path/to/noise/audio） | `None` |
| `--bg_noise_snr_min` | 背景噪声最小信噪比（dB） | `5` |
| `--bg_noise_snr_max` | 背景噪声最大信噪比（dB） | `20` |

**数据模拟流程：**

1. 添加背景噪声（可选，模拟麦克风拾取人声和背景噪音）
2. 下采样到 8kHz
3. 应用 300-3400 Hz 带通滤波器（模拟电话线路频率响应）
4. 应用 G.711 μ-law/A-law 编解码压扩
5. 添加电话线路噪声（白噪声 + 50/60Hz 工频干扰）

### 2. 模型微调训练

#### 2.1 配置训练脚本

编辑 `finetune_8k_telephone.sh`，修改以下关键参数：

```bash
# 输入数据（已模拟的 8kHz 数据）
train_data="/path/to/8khz/train.jsonl"
val_data="/path/to/8khz/val.jsonl"

# GPU 配置
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 训练超参数
batch_size=4
max_epoch=20
learning_rate=0.00005  # 5e-5
```

#### 2.2 启动训练

```bash
# 直接开始训练（假设数据已准备好）
bash finetune_8k_telephone.sh 1 1

# 完整流程（数据模拟 + 训练）
bash finetune_8k_telephone.sh -1 1
```

**训练阶段：**

- **Stage -1**：数据模拟（16kHz → 8kHz 电话信道）
- **Stage 0**：数据准备（验证数据文件，统计样本数）
- **Stage 1**：微调训练
- **Stage 2**：评估

#### 2.3 分布式训练

脚本自动支持多 GPU 分布式训练。对于更大规模训练，可启用 DeepSpeed：

```bash
# 修改 finetune_8k_telephone.sh
use_deepspeed=true
deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage1.json"
```

### 3. 模型推理

#### 3.1 单文件推理

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/test.wav \
    --output_dir ./output \
    --device cuda:0
```

#### 3.2 批量推理

```bash
# 准备音频文件列表
echo "/path/to/audio1.wav" > audio_list.txt
echo "/path/to/audio2.wav" >> audio_list.txt

python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_list audio_list.txt \
    --output_dir ./output \
    --batch_size 4 \
    --device cuda:0
```

#### 3.3 JSONL 评估

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --test_data /path/to/test.jsonl \
    --output_dir ./output \
    --batch_size 4 \
    --audio_key source \
    --text_key target \
    --device cuda:0
```

输出结果包含：
- `results.jsonl`：每个样本的识别结果和 CER/WER
- `metrics.json`：整体评估指标（平均 CER/WER、RTF 等）

#### 3.4 热词增强

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/test.wav \
    --output_dir ./output \
    --hotwords "客服中心,订单号,退款" \
    --device cuda:0
```

#### 3.5 VAD 长音频分段

```bash
python inference_8k.py \
    --model_dir /path/to/exp/output_dir \
    --checkpoint /path/to/exp/output_dir/model.pt.ep10 \
    --audio_file /path/to/long_audio.wav \
    --output_dir ./output \
    --use_vad \
    --vad_model fsmn-vad \
    --max_single_segment_time 30000 \
    --device cuda:0
```

## 训练配置说明

配置文件位于 `conf/config_8k_telephone.yaml`，关键配置项说明如下：

### Frontend 配置（8kHz 适配）

```yaml
frontend_conf:
    fs: 8000          # 8kHz 采样率
    n_fft: 200        # 25ms 窗口（200 samples @ 8kHz）
    hop_length: 80    # 10ms 步长（80 samples @ 8kHz）
    n_mels: 80        # Mel 滤波器组数量
```

### Audio Encoder 配置（部分微调）

```yaml
audio_encoder_conf:
    freeze: false           # 允许微调
    freeze_layer_num: 12    # 冻结前 12 层，微调后面层
```

**设计思路：**
- 前 12 层保留通用特征提取能力
- 后面层适配 8kHz 窄带信号特征

### LLM 配置（LoRA 微调）

```yaml
llm_conf:
    freeze: true      # 冻结基础 LLM
    use_lora: true    # 启用 LoRA 高效微调
    lora_conf:
        r: 32                 # LoRA 秩
        lora_alpha: 64        # LoRA 缩放因子
        target_modules:       # 目标模块
            - "q_proj"
            - "v_proj"
            - "k_proj"
            - "o_proj"
            - "gate_proj"
            - "up_proj"
            - "down_proj"
```

**LoRA 优势：**
- 大幅减少可训练参数（通常 < 1% 原模型参数）
- 加速训练，降低显存占用
- 保护预训练知识，防止灾难性遗忘

### Audio Adaptor 配置（全量微调）

```yaml
audio_adaptor: Linear
audio_adaptor_conf:
    downsample_rate: 5
    llm_dim: 1536      # Qwen2.5-1.5B 隐藏层维度
    encoder_dim: 512   # 音频编码器输出维度
```

### SpecAug 配置（8kHz 优化）

```yaml
specaug_conf:
    freq_mask_width_range: [0, 15]  # 减小频率掩码范围（8kHz 频谱更窄）
    time_mask_width_range: [0, 10]  # 适当减小时间掩码范围
```

### 优化器配置

```yaml
optim: adamw
optim_conf:
    lr: 0.00005         # 5e-5，较低学习率保护预训练知识
    weight_decay: 0.01
```

## 性能优化建议

### 1. 数据增强

- 使用不同 SNR 范围（10-30 dB）生成多样化训练数据
- 混合 μ-law 和 A-law 编解码模拟
- 添加真实电话录音作为补充训练数据

### 2. 训练策略

- **预热策略**：前 1000 步使用较小学习率预热
- **学习率调度**：使用 warmup + cosine decay
- **梯度累积**：显存不足时增大 `accum_grad`（如设为 2 或 4）

### 3. 超参数调优

| 超参数 | 推荐范围 | 说明 |
|--------|----------|------|
| 学习率 | 3e-5 ~ 1e-4 | 较低学习率保护预训练知识 |
| Batch Size | 2 ~ 8 | 根据 GPU 显存调整 |
| LoRA Rank | 16 ~ 64 | 越大表达能力越强，但训练成本越高 |
| 冻结层数 | 8 ~ 16 | 越多保留越多通用特征 |

### 4. 显存优化

```bash
# 启用 DeepSpeed ZeRO Stage 2
use_deepspeed=true
deepspeed_config="${workspace}/../../deepspeed_conf/ds_stage2.json"

# 减小 batch size，增大梯度累积
batch_size=2
accum_grad=4
```

## 常见问题

### Q1: 训练数据量需要多少？

**建议：**
- 最少：1000 小时 16kHz 高质量数据 → 模拟为 8kHz
- 推荐：5000+ 小时（覆盖多种口音、噪声场景）

### Q2: 如何处理真实电话录音？

真实电话录音通常已是 8kHz，可以直接用于微调，建议：
1. 跳过数据模拟步骤
2. 确保音频采样率为 8000 Hz
3. 混合真实录音和模拟数据训练

### Q3: 训练多久会收敛？

根据数据量和 GPU 配置：
- 小规模（1000 小时）：4-8 个 epoch，约 1-2 天（4x V100）
- 大规模（10000 小时）：10-15 个 epoch，约 5-7 天（8x A100）

### Q4: CER/WER 指标预期是多少？

参考指标（取决于数据质量和场景）：
- 清晰电话录音（SNR > 20dB）：CER < 5%, WER < 10%
- 一般电话录音（SNR 15-20dB）：CER 8-12%, WER 15-20%
- 噪声电话录音（SNR < 15dB）：CER 15-25%, WER 25-35%

### Q5: 如何进一步提升准确率？

1. **数据质量**：
   - 清洗错误标注
   - 增加领域相关训练数据
   - 平衡不同口音、噪声场景

2. **模型优化**：
   - 增大 LoRA rank（如 64）
   - 解冻更多 encoder 层
   - 增加训练 epoch

3. **推理优化**：
   - 使用热词增强
   - 结合语言模型 rescoring
   - 后处理规则优化

## 技术细节

### 电话信道模拟原理

#### 1. 带通滤波器

电话线路频率响应为 300-3400 Hz，使用 4 阶巴特沃斯带通滤波器模拟：

```python
low = 300 / (fs / 2)   # 归一化低频
high = 3400 / (fs / 2) # 归一化高频
b, a = signal.butter(4, [low, high], btype='band')
filtered = signal.filtfilt(b, a, audio)
```

#### 2. G.711 μ-law 压扩

μ-law 压扩公式（μ = 255）：

```
压缩：F(x) = sign(x) * log(1 + μ|x|) / log(1 + μ)
解压：F⁻¹(y) = sign(y) * (1/μ) * [(1 + μ)^|y| - 1]
```

量化到 8-bit（256 级）模拟 G.711 编解码。

#### 3. 噪声模型

- **白噪声**：高斯白噪声，功率为噪声总功率的 80%
- **工频干扰**：50/60 Hz 正弦波及其二次谐波，功率为噪声总功率的 20%

```python
white_noise = np.random.randn(len(audio)) * sqrt(0.8 * noise_power)
power_line = sqrt(0.2 * noise_power) * sin(2π * 50 * t)
```

### LoRA 微调原理

LoRA（Low-Rank Adaptation）通过低秩矩阵分解实现高效微调：

```
W' = W + ΔW = W + BA
```

其中：
- W：预训练权重（冻结）
- B：r × d 矩阵
- A：d × r 矩阵
- r：秩（r << d）

**参数量对比：**
- 原模型：d × d
- LoRA：2 × d × r（r << d 时大幅减少）

例如：d = 4096, r = 32
- 原模型：16,777,216 参数
- LoRA：262,144 参数（减少 98.4%）

## 引用

如果本方案对您的研究或项目有帮助，请引用：

```bibtex
@misc{funasr_8k_telephone,
    title={Fun-ASR-Nano 8kHz Telephone Channel Fine-tuning Scheme},
    author={FunASR Team},
    year={2025},
    howpublished={\url{https://github.com/modelscope/FunASR}},
}
```

## 许可证

本项目遵循 [MIT License](../../../LICENSE)。

## 联系方式

- GitHub Issues: https://github.com/modelscope/FunASR/issues
- ModelScope: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512

## 更新日志

- **2025-01**: 初始版本发布
  - 数据模拟工具
  - 8kHz 训练配置
  - 推理评估脚本
  - 完整文档
