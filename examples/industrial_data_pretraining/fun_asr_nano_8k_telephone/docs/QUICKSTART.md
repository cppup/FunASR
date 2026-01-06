# Fun-ASR-Nano 8kHz 电话微调 - 快速开始

## 前提条件

1. **环境安装**
```bash
cd /workspace/share/LLMFunASR
pip install -r requirements.txt
```

2. **下载预训练模型**
```bash
python -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-ASR-Nano-2512')"
```

---

## 数据准备

### 方案A: 使用现有真实 8kHz 数据

如果你已有真实 8kHz 电话数据，需要：

1. **准备简化格式 JSONL**
```json
{"source": "/path/to/audio.wav", "target": "转写文本", "key": "sample_001"}
```

2. **转换为训练格式（上采样 + 添加 messages）**
```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

python data/prepare_training_data.py convert \
    --input /path/to/raw_8k.jsonl \
    --output /output/funasr/data/real_8k_telephone/train_50h.jsonl \
    --output_audio_dir /output/funasr/data/real_8k_telephone/audio_16k \
    --do_upsample \
    --target_fs 16000 \
    --num_workers 32
```

3. **验证数据**
```bash
python data/prepare_training_data.py validate \
    --input /output/funasr/data/real_8k_telephone/train_50h.jsonl \
    --check_audio
```

---

### 方案B: 从 16kHz 数据模拟电话信道

如果只有 16kHz 高质量数据，需要模拟电话效果：

1. **准备 16kHz 数据（JSONL 格式）**
```json
{"source": "/path/to/16k_audio.wav", "target": "转写文本"}
```

2. **运行信道模拟（增强版，包含 VoIP 丢帧、Babble Noise）**
```bash
python data/data_simulation.py \
    --input /path/to/16k_train.jsonl \
    --output /output/funasr/data/simulated_8k_telephone/train_10k.jsonl \
    --output_audio_dir /output/funasr/data/simulated_8k_telephone/audio \
    --audio_key source \
    --target_fs 8000 \
    --output_fs 16000 \
    --low_freq 300 \
    --high_freq 3400 \
    --codec_type mu-law \
    --snr_db_min 10 \
    --snr_db_max 25 \
    --num_workers 64
```

**模拟效果包含**:
- ✅ 8kHz 采样（频谱截断）
- ✅ G.711 μ-law 编解码失真
- ✅ 300-3400Hz 带通滤波
- ✅ VoIP 丢帧模拟（随机 + 连续丢帧）
- ✅ Babble Noise（背景人声，5-15dB SNR）
- ✅ 白噪声 + 工频干扰

3. **转换为训练格式**
```bash
python data/prepare_training_data.py convert \
    --input /output/funasr/data/simulated_8k_telephone/train_10k.jsonl \
    --output /output/funasr/data/simulated_8k_telephone/train_10k_formatted.jsonl \
    --num_workers 32
```

---

## 三阶段微调

### Stage 1: Encoder 适配 (1000h 模拟数据)

```bash
bash finetune_3stage.sh 1 1
```

**训练内容**: 解冻 Audio Encoder 后 20 层，适应 8kHz 频谱  
**数据**: 1000h+ 模拟电话数据  
**时间**: ~2 天 (4x V100)  
**Checkpoint**: `/output/funasr/exp/8k_telephone/stage1_encoder_*/model.pt.avg`

---

### Stage 2: Adapter 对齐 (1000h 模拟数据)

```bash
bash finetune_3stage.sh 2 2
```

**训练内容**: 微调 Adapter + CTC Decoder，重新对齐  
**数据**: 1000h+ 模拟电话数据  
**时间**: ~2 天 (4x V100)  
**Checkpoint**: `/output/funasr/exp/8k_telephone/stage2_adapter_*/model.pt.avg`

---

### Stage 3: LoRA 业务适配 (50h 真实数据)

```bash
bash finetune_3stage.sh 3 3
```

**训练内容**: LLM LoRA 微调，注入业务术语  
**数据**: 50h 真实电话外呼数据  
**时间**: ~1 天 (4x V100)  
**Final Checkpoint**: `/output/funasr/exp/8k_telephone/stage3_lora_*/model.pt.avg`

---

## 完整三阶段流程（一键执行）

```bash
# 确保数据已准备好
export SIMULATED_DATA="/output/funasr/data/simulated_8k_telephone"
export REAL_DATA="/output/funasr/data/real_8k_telephone"

# 运行三阶段
bash finetune_3stage.sh 1 3
```

**总耗时**: ~5 天 (4x V100)

---

## 评估

### 1. 标准评估（CER/WER）

```bash
python inference_8k.py \
    --model_dir /output/funasr/exp/8k_telephone/stage3_lora_* \
    --checkpoint /output/funasr/exp/8k_telephone/final_checkpoint.txt \
    --test_data /path/to/test.jsonl \
    --output_dir ./evaluation_results \
    --batch_size 4
```

### 2. 关键词评估（KWER）

```bash
python evaluate_keywords.py \
    --results ./evaluation_results/results.jsonl \
    --keywords keywords_example.txt \
    --output ./evaluation_results/kwer_metrics.json
```

**输出示例**:
```
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
  ...
```

---

## 常见问题

### Q1: 没有 1000h 模拟数据怎么办？

**A**: 可以跳过 Stage 1-2，直接从预训练模型开始 Stage 3:

```bash
# 修改 finetune_3stage.sh 中的 stage2_ckpt
# 指向预训练模型: FunAudioLLM/Fun-ASR-Nano-2512/model.pt

bash finetune_3stage.sh 3 3
```

但效果会打折扣，建议至少准备 500h 模拟数据。

### Q2: Stage 1 训练很慢？

**A**: 可以减少 Encoder 微调层数:

```yaml
# stage1_encoder_adapt.yaml
audio_encoder_conf:
    freeze_layer_num: 40  # 只微调后10层
```

### Q3: 如何监控训练效果？

**A**: 查看 log 和 tensorboard:

```bash
# 查看训练日志
tail -f /output/funasr/exp/8k_telephone/stage1_*/train.log

# 启动 tensorboard
tensorboard --logdir /output/funasr/exp/8k_telephone/ --port 6006
```

---

## 下一步

1. **优化关键词识别**: 根据 KWER 报告，针对性添加热词
2. **多场景测试**: 在不同噪声、口音场景下评估
3. **部署生产**: 导出模型到推理服务
