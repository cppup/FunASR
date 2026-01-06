# FunASR-Nano 8kHz 电话语音微调 - 实施方案

> **版本**: v2.1 (2025-01-05)  
> **目标**: 将 Fun-ASR-Nano 适配到 8kHz 中文电话语音识别场景

---

## 一、现状分析（基于代码验证）

### 1. FunASR-Nano 模型架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       Fun-ASR-Nano-2512                         │
├─────────────────────────────────────────────────────────────────┤
│  Input: Audio (16kHz) + Text Prompt                             │
│                    ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  WavFrontend (fs=16000, n_mels=80, lfr_m=7, lfr_n=6)    │   │
│  │  - 输出: Fbank 特征 [B, T, 80]                          │   │
│  │  - 下采样: 帧移10ms × LFR_n=6 → 60ms/帧                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SenseVoiceEncoderSmall (50 blocks, output_size=512)    │   │
│  │  - 2层CNN子采样: T_out ≈ T / 4                          │   │
│  │  - 输出: [B, T_out, 512]                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Audio Adaptor (Transformer, 2 layers)                  │   │
│  │  - encoder_dim: 512 → llm_dim: 1024                    │   │
│  │  - 输出: [B, T_adaptor, 1024]                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    ↓                                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Qwen3-0.6B LLM (hidden_dim=1024)                       │   │
│  │  - 生成文本输出                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                    ↓                                            │
│  Output: Transcription Text                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2. ⚠️ 重要代码限制（已验证）

经代码审查，`funasr/models/fun_asr_nano/model.py` 中：

```python
# 第61-66行：freeze_layer_num 被读取但未实现分层冻结逻辑！
freeze_layer_num = int(audio_encoder_conf.get("freeze_layer_num", -1))

if freeze:  # 如果 freeze=True，则全部冻结
    for name, param in audio_encoder.named_parameters():
        param.requires_grad = False  # 没有根据 layer_id 判断
    audio_encoder.eval()
```

**结论**: `freeze_layer_num` 配置在 `FunASRNano` 模型中**不生效**！

**解决方案**: 
- 方案A: 使用 `freeze: false` 全量微调 Encoder
- 方案B: 修改模型代码添加分层冻结支持（见附录）

### 3. 问题识别
- **问题1**: 8kHz 数据必须上采样到 16kHz 才能输入 WavFrontend
- **问题2**: 电话信道失真需要模型适应
- **问题3**: `freeze_layer_num` 不生效，需替代方案

---

## 二、三阶段微调策略（已验证可落地）

### 数据流程图

```
16kHz 高质量语料（开源 WenetSpeech 等）
    ↓ 
[data_simulation.py] 1000h+ 模拟数据
 ├─ 降采样 8kHz
 ├─ G.711 编解码（μ-law/A-law）
 ├─ 带通滤波（300-3400Hz）
 ├─ VoIP 丢帧模拟（1-5%）
 └─ Babble Noise（5-15dB SNR）
    ↓ 上采样 16kHz（保留失真特征）
    ↓
[prepare_training_data.py convert] 转换为 messages 格式
    ↓
[Stage 1] Audio Encoder 全量微调
    → 学习 8kHz 频谱特征
    ↓
[Stage 2] Adapter + CTC Decoder 微调
    → 重新映射到 Token 空间
    ↓
50h 真实电话外呼数据
    ↓
[Stage 3] LLM LoRA 业务适配
    → 注入业务术语
```

---

### Stage 1: Audio Encoder 适配 8kHz

**目标**: 让 Encoder 适应 8kHz 频谱分布（高频缺失、编解码失真）

| 组件 | 策略 | 可训练参数 | 备注 |
|------|------|-----------|------|
| WavFrontend | 冻结 | 0 | 无可训练参数 |
| **Audio Encoder** | **全量微调** | **~150M** | freeze=false |
| Audio Adaptor | 冻结 | 0 | freeze=true |
| CTC Decoder | 冻结 | 0 | freeze=true |
| LLM | 冻结 | 0 | freeze=true |

**训练配置**:

| 项目 | 配置 |
|------|------|
| 数据量 | 1000h+ 模拟电话数据 |
| 学习率 | **1e-4** (较大，快速适应) |
| Epoch | 5-10 |
| Batch Size | 8192 (token-based) |
| SpecAugment time_mask | 50 帧（模拟 VoIP 丢帧） |
| 预训练初始化 | `FunAudioLLM/Fun-ASR-Nano-2512/model.pt` |

**配置文件**: `conf/stage1_encoder_adapt.yaml`

**预期**: 5 epochs 后 loss 下降 30%+

---

### Stage 2: Adapter & CTC Decoder 对齐

**目标**: 重新对齐 8kHz 音频特征到 LLM Token 空间

| 组件 | 策略 | 可训练参数 | 备注 |
|------|------|-----------|------|
| WavFrontend | 冻结 | 0 | - |
| Audio Encoder | 冻结 | 0 | 使用 Stage 1 权重 |
| **Audio Adaptor** | **全量微调** | **~4M** | 2层 Transformer |
| **CTC Decoder** | **全量微调** | **~8M** | 5层 Transformer |
| LLM | 冻结 | 0 | - |

**训练配置**:

| 项目 | 配置 |
|------|------|
| 数据量 | 1000h+ 模拟电话数据 |
| 学习率 | **5e-5** (中等) |
| Epoch | 10 |
| Batch Size | 8192 |
| 初始化 | Stage 1 checkpoint |

**配置文件**: `conf/stage2_adapter_align.yaml`

**预期**: CTC Loss 下降，WER 开始收敛

---

### Stage 3: LLM LoRA 业务语义注入

**目标**: 注入业务术语（产品名、意图词）和电话口语习惯

| 组件 | 策略 | 可训练参数 | 备注 |
|------|------|-----------|------|
| Audio Encoder | 冻结 | 0 | - |
| Audio Adaptor | 冻结 | 0 | - |
| CTC Decoder | 冻结 | 0 | - |
| **LLM** | **LoRA** | **~2M** | r=16, α=32 |

**LoRA 目标模块**: `q_proj`, `v_proj`, `k_proj`, `o_proj`

**训练配置**:

| 项目 | 配置 |
|------|------|
| 数据量 | 50h 真实电话外呼数据 |
| 学习率 | **1e-5** (小，防过拟合) |
| Epoch | 20 |
| Batch Size | 4096 |
| Gradient Accum | 2 |
| 初始化 | Stage 2 checkpoint |

**配置文件**: `conf/stage3_lora_domain.yaml`

**预期**: 业务关键词准确率（KWER）提升 10%+

---

### 三阶段总结表

| 阶段 | 数据 | 训练模块 | 冻结模块 | 可训练参数 | LR | Epochs |
|------|------|----------|----------|-----------|-----|--------|
| **S1** | 1000h 模拟 | Encoder | Adaptor, CTC, LLM | ~150M | 1e-4 | 5-10 |
| **S2** | 1000h 模拟 | Adaptor, CTC | Encoder, LLM | ~12M | 5e-5 | 10 |
| **S3** | 50h 真实 | LLM (LoRA) | Encoder, Adaptor, CTC | ~2M | 1e-5 | 20 |

**总训练时间** (4× A100-80G):
- Stage 1: ~3 天
- Stage 2: ~2 天  
- Stage 3: ~1 天
- **总计**: ~6 天

---

## 三、数据准备流程（完整流水线）

### 3.1 数据流水线图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           完整数据流水线                                    │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Step 1: 准备 16kHz 源数据（简化格式）                                      │
│  ────────────────────────────────────                                      │
│  {"source": "/path/to/16k.wav", "target": "转写文本"}                       │
│                            ↓                                               │
│  Step 2: 电话信道模拟 (data_simulation.py)                                  │
│  ────────────────────────────────────────                                  │
│  - 降采样 16kHz → 8kHz                                                     │
│  - G.711 编解码失真                                                        │
│  - 300-3400Hz 带通滤波                                                     │
│  - VoIP 丢帧模拟 (1-5%)                                                    │
│  - Babble Noise (5-15dB)                                                   │
│  - 上采样 8kHz → 16kHz（保留失真）                                          │
│                            ↓                                               │
│  输出: {"source": "/path/to/sim_16k.wav", "target": "转写文本"}             │
│                            ↓                                               │
│  Step 3: 格式转换 (prepare_training_data.py convert)                       │
│  ────────────────────────────────────────────────                          │
│  输出: messages 格式 JSONL                                                 │
│  {                                                                         │
│    "key": "sample_001",                                                    │
│    "messages": [                                                           │
│      {"role": "system", "content": "You are a helpful assistant."},        │
│      {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},│
│      {"role": "assistant", "content": "转写文本"}                          │
│    ],                                                                      │
│    "speech_length": 800,                                                   │
│    "text_length": 15                                                       │
│  }                                                                         │
│                            ↓                                               │
│  Step 4: 数据验证 (prepare_training_data.py validate)                      │
│  ─────────────────────────────────────────────────                         │
│  - 检查 messages 结构                                                      │
│  - 检查音频路径有效性                                                       │
│  - 统计 speech_length / text_length                                        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 执行命令

```bash
# ========================================
# Step 1: 准备 16kHz 源数据
# ========================================
# 格式: {"source": "/path/to/audio.wav", "target": "转写文本"}
# 支持 WenetSpeech、AISHELL 等数据集转换

# ========================================
# Step 2: 电话信道模拟
# ========================================
python data/data_simulation.py \
    --input /path/to/16k_train.jsonl \
    --output /output/simulated/train_simulated.jsonl \
    --output_audio_dir /output/simulated/audio \
    --audio_key source \
    --target_fs 8000 \
    --output_fs 16000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 10 \
    --snr_db_max 25 \
    --num_workers 64

# ========================================
# Step 3: 格式转换
# ========================================
python data/prepare_training_data.py convert \
    --input /output/simulated/train_simulated.jsonl \
    --output /output/simulated/train_formatted.jsonl \
    --task_template "语音转写：" \
    --num_workers 32

# ========================================
# Step 4: 数据验证
# ========================================
python data/prepare_training_data.py validate \
    --input /output/simulated/train_formatted.jsonl \
    --check_audio

# ========================================
# 或者一键执行 (finetune_3stage.sh Stage 0)
# ========================================
export RAW_16K_TRAIN=/path/to/16k_train.jsonl
export RAW_16K_VAL=/path/to/16k_val.jsonl
bash finetune_3stage.sh 0 0
```

### 3.3 真实 8kHz 数据准备

```bash
# 如果已有真实 8kHz 电话录音
python data/prepare_training_data.py convert \
    --input /path/to/real_8k.jsonl \
    --output /output/real/train_formatted.jsonl \
    --output_audio_dir /output/real/audio_16k \
    --do_upsample \
    --target_fs 16000 \
    --num_workers 32
```

---

## 四、训练执行

### 4.1 环境变量配置

```bash
# 必须配置
export CUDA_VISIBLE_DEVICES=0,1,2,3          # GPU 设备
export OUTPUT_ROOT=/output/funasr            # 输出目录
export MODELSCOPE_CACHE=~/.cache/modelscope  # 模型缓存

# 数据路径 (Stage 0 会生成)
export RAW_16K_TRAIN=/path/to/16k_train.jsonl
export RAW_16K_VAL=/path/to/16k_val.jsonl

# Stage 3 真实数据
export REAL_TRAIN=/path/to/real_8k_train.jsonl
export REAL_VAL=/path/to/real_8k_val.jsonl
```

### 4.2 分阶段执行

```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# Stage 0: 数据准备 (可选，如已有数据可跳过)
bash finetune_3stage.sh 0 0

# Stage 1: Encoder 适配
bash finetune_3stage.sh 1 1

# Stage 2: Adapter 对齐
bash finetune_3stage.sh 2 2

# Stage 3: LoRA 业务适配
bash finetune_3stage.sh 3 3

# 或一键执行全部
bash finetune_3stage.sh 0 3
```

### 4.3 Checkpoint 链

```
PTM (Fun-ASR-Nano-2512)
    ↓ init_param
Stage 1 checkpoint: ${OUTPUT_ROOT}/exp/8k_telephone/stage1_*/model.pt.avg
    ↓ init_param
Stage 2 checkpoint: ${OUTPUT_ROOT}/exp/8k_telephone/stage2_*/model.pt.avg
    ↓ init_param
Stage 3 checkpoint: ${OUTPUT_ROOT}/exp/8k_telephone/stage3_*/model.pt.avg (Final)
```

---

## 五、评估与验证

### 5.1 推理测试

```bash
python inference_8k.py \
    --model_path ${OUTPUT_ROOT}/exp/8k_telephone/final_checkpoint.txt \
    --test_data /path/to/test.jsonl \
    --output_dir ./results \
    --device cuda:0
```

### 5.2 关键词评估（KWER）

```bash
# 使用业务关键词文件
python evaluate_keywords.py \
    --results ./results/results.jsonl \
    --keywords keywords_example.txt \
    --output ./results/kwer_report.json
```

**输出示例**:
```
================================================================================
ASR Evaluation Report
================================================================================
Total samples: 500
Valid samples: 500

Overall Metrics:
  Character Error Rate (CER): 8.50%
  Word Error Rate (WER): 12.30%
  Keyword Error Rate (KWER): 5.20%

Per-Keyword Accuracy:
  Keyword              Accuracy    Total  Correct    Error
  -------------------- ---------- -------- -------- --------
  办理                    95.00%       20       19        1
  同意                    92.50%       40       37        3
  订单号                  88.00%       25       22        3
================================================================================
```

### 5.3 训练监控检查清单

**训练前**:
- [ ] `frontend_conf.fs = 16000` 确认
- [ ] 数据已转换为 messages 格式
- [ ] 预训练模型已下载
- [ ] GPU 显存足够 (建议 4×A100-80G 或 8×V100-32G)

**训练中**:
- [ ] Loss 正常下降
- [ ] 无 OOM 错误
- [ ] Checkpoint 按时保存

**训练后**:
- [ ] CER/WER 达标 (<10%)
- [ ] KWER 提升 (比 baseline 提升 10%+)
- [ ] 无幻觉/复读现象

---

## 六、故障排查

### 问题1: 预训练模型加载失败
```
Error: model path not found / FileNotFoundError
```
**解决**: 下载预训练模型
```bash
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512')"
```

### 问题2: 音频采样率不匹配
```
Error: Sample rate mismatch / Frontend expects 16000 but got 8000
```
**原因**: 输入音频未上采样到 16kHz  
**解决**: 
```bash
python data/prepare_training_data.py convert --do_upsample --target_fs 16000
```

### 问题3: OOM (显存不足)
```
RuntimeError: CUDA out of memory
```
**解决**: 减小 batch_size 或启用 gradient checkpointing
```bash
++dataset_conf.batch_size=4096
++train_conf.accum_grad=2
```

### 问题4: Loss 不下降
**可能原因**:
1. 学习率过小 → 增大 lr
2. 数据格式错误 → 检查 messages 结构
3. 未加载预训练权重 → 确认 init_param 路径

### 问题5: freeze_layer_num 不生效
**原因**: `FunASRNano` 模型代码未实现分层冻结  
**解决**: 使用 `freeze: false` 全量微调，或修改模型代码（见附录 A）

---

## 七、文件清单

```
fun_asr_nano_8k_telephone/
├── IMPLEMENTATION_PLAN.md          # 本实施方案 ★
├── TRAINING_DATA_SPEC.md           # 训练数据格式规范
├── QUICKSTART.md                   # 快速开始指南
├── README.md / README_zh.md        # 项目说明
├── requirements.txt                # Python 依赖
├── keywords_example.txt            # 示例业务关键词
│
├── conf/                           # 训练配置
│   ├── stage1_encoder_adapt.yaml   # Stage 1: Encoder 全量微调
│   ├── stage2_adapter_align.yaml   # Stage 2: Adapter + CTC 微调
│   ├── stage3_lora_domain.yaml     # Stage 3: LLM LoRA 微调
│   ├── config_8k_telephone.yaml    # 单阶段配置 (备用)
│   └── config_8k_telephone_from_hub.yaml
│
├── data/                           # 数据处理工具
│   ├── data_simulation.py          # 电话信道模拟
│   └── prepare_training_data.py    # 格式转换 + 验证
│
├── finetune_3stage.sh              # 三阶段训练脚本 ★
├── finetune_8k_telephone.sh        # 单阶段训练脚本 (备用)
├── inference_8k.py                 # 推理脚本
└── evaluate_keywords.py            # KWER 评估脚本
```

---

## 八、附录

### 附录 A: 如需分层冻结 Encoder

如果需要实现 `freeze_layer_num` 功能，需修改 `funasr/models/fun_asr_nano/model.py`:

```python
# 第 61-66 行，将：
if freeze:
    for name, param in audio_encoder.named_parameters():
        param.requires_grad = False
    audio_encoder.eval()

# 改为：
if freeze:
    for name, param in audio_encoder.named_parameters():
        if freeze_layer_num > 0:
            import re
            idx = re.search(r"\.\d+\.", name)
            if idx is not None:
                beg, end = idx.regs[0]
                layer_id = int(name[beg + 1 : end - 1])
                if layer_id < freeze_layer_num:
                    param.requires_grad = False
                # else: 保持 requires_grad=True (微调)
            elif "ln_post." not in name:
                param.requires_grad = False
        else:
            param.requires_grad = False
    audio_encoder.eval()
```

### 附录 B: 参考资源

- **模型**: [Fun-ASR-Nano-2512 (ModelScope)](https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512)
- **Demo**: [ModelScope Space](https://modelscope.cn/studios/FunAudioLLM/Fun-ASR-Nano)
- **FunASR**: [GitHub](https://github.com/modelscope/FunASR)

---

> **版本历史**
> - v2.1 (2025-01-05): 修复 freeze_layer_num 不生效问题，完善数据流水线，添加完整执行命令
> - v2.0 (2025-01-05): 三阶段微调策略
> - v1.0 (2025-01-04): 初始版本
