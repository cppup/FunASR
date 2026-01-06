# 三阶段微调实验执行计划

> **实验启动时间**: 2026-01-05  
> **执行人**: Algorithm Research Team  
> **目标**: 将 Fun-ASR-Nano 适配到 8kHz 中文电话语音识别场景

---

## 一、实验环境配置 ✅

### 硬件资源
- **GPU**: 8× NVIDIA H100 80GB (使用 GPU 2,3)
- **GPU 内存**: 81GB × 2 = 162GB 可用
- **计算卡**: GPU 2, 3 (避免与其他任务冲突)

### 软件环境
- **FunASR**: v1.2.9 ✅
- **Python**: 3.x ✅
- **PyTorch**: 已安装 ✅
- **依赖**: 已安装 ✅

### 工作目录
```
/workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/
├── run_experiment.sh          # 主实验脚本 ✅
├── test_data_prep.sh          # 数据准备测试 ✅
├── conf/
│   ├── stage1_encoder_adapt.yaml    # Stage 1 配置 ✅
│   ├── stage2_adapter_align.yaml    # Stage 2 配置 ✅
│   └── stage3_lora_domain.yaml      # Stage 3 配置 ✅
├── data/
│   ├── data_simulation.py           # 电话信道模拟 ✅
│   └── prepare_training_data.py     # 数据格式转换 ✅ (已添加 filter_annotation_error)
└── exp_output/                      # 输出目录
```

---

## 二、数据准备 ✅

### Stage 1-2 训练数据: WenetSpeech (1000h)
- **路径**: `/data/speech/open/data/openslr/chinese/WenetSpeech/jsonl/funasr_jsonl/`
- **训练集**: `train_M.jsonl` (65,170 samples)
- **验证集**: `eval_dev.jsonl` (13,825 samples)
- **格式**: 已切分的 16kHz WAV + JSONL

### Stage 3 训练数据: 电话外呼真实数据 (50h)
- **路径**: `/data/speech/labeled/yx_telecall/sale/training_data/yx_telecall_v2_1_2025-12-22/manifests/`
- **训练集**: `audio_neutral.jsonl` (12,172 samples)
- **验证集**: `audio_quiet.jsonl` (9,001 samples)
- **格式**: 8kHz 电话录音 + JSONL
- **质量**: 包含 `<ANNOTATION_ERROR>` 标记，需过滤

### 数据增强策略
WenetSpeech → 电话信道模拟:
1. 降采样到 8kHz
2. G.711 μ-law 编解码
3. 300-3400Hz 带通滤波
4. VoIP 丢帧模拟 (1-2%)
5. Babble Noise (10-25dB SNR)
6. 上采样回 16kHz (保留失真)

---

## 三、实验执行步骤

### 🔧 Step 0: 数据准备测试 (预计 10 分钟)

```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# 运行小规模测试 (10 samples)
bash test_data_prep.sh
```

**验证点**:
- ✓ WenetSpeech 数据可读
- ✓ 电话数据可读
- ✓ 模拟脚本正常运行
- ✓ 格式转换正常
- ✓ 数据验证通过

---

### 📊 Stage 0: 完整数据准备 (预计 4-6 小时)

```bash
# 模拟 WenetSpeech → 8kHz 电话音频 (65k samples)
bash run_experiment.sh 0 0
```

**输出**:
- `exp_output/data/simulated_8k_telephone/train_formatted.jsonl` (~65k samples)
- `exp_output/data/simulated_8k_telephone/dev_formatted.jsonl` (~14k samples)
- `exp_output/data/real_8k_telephone/train_formatted.jsonl` (~10k samples, 过滤后)
- `exp_output/data/real_8k_telephone/dev_formatted.jsonl` (~9k samples, 过滤后)

**检查点**:
```bash
# 验证数据量
wc -l exp_output/data/simulated_8k_telephone/*.jsonl
wc -l exp_output/data/real_8k_telephone/*.jsonl

# 检查样本
head -1 exp_output/data/simulated_8k_telephone/train_formatted.jsonl | python -m json.tool
```

---

### 🎯 Stage 1: Audio Encoder 适配 (预计 2-3 天)

**目标**: 让 Encoder 适应 8kHz 频谱特征

```bash
bash run_experiment.sh 1 1
```

**训练配置**:
- **可训练参数**: Audio Encoder (~150M)
- **冻结模块**: Adaptor, CTC, LLM
- **学习率**: 1e-4
- **Batch size**: 8192 tokens
- **Epoch**: 10
- **GPU**: 2 × H100

**监控指标**:
```bash
# 查看训练日志
tail -f exp_output/exp/8k_telephone/stage1_encoder_*/train.log

# 关键指标
# - Loss 下降 30%+
# - CER 在验证集上改善
```

**预期效果**:
- ✓ Loss 从 ~8.0 降至 ~5.5
- ✓ Encoder 学到 8kHz 频谱特征
- ✓ Checkpoint 保存到 `exp_output/exp/8k_telephone/stage1_checkpoint.txt`

---

### 🔄 Stage 2: Adapter & CTC 对齐 (预计 2-3 天)

**目标**: 重新对齐 8kHz 特征到 Token 空间

```bash
bash run_experiment.sh 2 2
```

**训练配置**:
- **可训练参数**: Adaptor (~4M) + CTC (~8M)
- **冻结模块**: Encoder (Stage 1), LLM
- **学习率**: 5e-5
- **Batch size**: 8192 tokens
- **Epoch**: 10

**监控指标**:
```bash
tail -f exp_output/exp/8k_telephone/stage2_adapter_*/train.log

# 关键指标
# - CTC Loss 显著下降
# - CER 进一步改善
```

**预期效果**:
- ✓ CTC Loss 下降 40%+
- ✓ CER < 10% (验证集)
- ✓ Checkpoint 保存到 `exp_output/exp/8k_telephone/stage2_checkpoint.txt`

---

### 🚀 Stage 3: LLM LoRA 业务适配 (预计 1-2 天)

**目标**: 注入电话外呼业务术语

```bash
bash run_experiment.sh 3 3
```

**训练配置**:
- **可训练参数**: LLM LoRA (r=16, ~2M)
- **冻结模块**: Encoder, Adaptor, CTC
- **学习率**: 1e-5
- **Batch size**: 4096 tokens
- **Epoch**: 20
- **数据**: 真实电话外呼数据 (~10k samples)

**监控指标**:
```bash
tail -f exp_output/exp/8k_telephone/stage3_lora_*/train.log

# 关键指标
# - 在业务术语上的准确率提升
# - KWER (关键词错误率) 改善
```

**预期效果**:
- ✓ 业务关键词准确率 > 95%
- ✓ 整体 CER < 8%
- ✓ 最终模型保存到 `exp_output/exp/8k_telephone/final_checkpoint.txt`

---

## 四、实验评估

### 评估脚本

```bash
# 1. 生成识别结果
python inference_8k.py \
    --model_path $(cat exp_output/exp/8k_telephone/final_checkpoint.txt) \
    --test_data /path/to/test.jsonl \
    --output_dir ./evaluation_results

# 2. 计算 KWER
python evaluate_keywords.py \
    --results ./evaluation_results/results.jsonl \
    --keywords keywords_example.txt \
    --output ./evaluation_results/kwer_metrics.json
```

### 评估指标

| 指标 | Baseline (16kHz) | Stage 1 | Stage 2 | Stage 3 (目标) |
|------|------------------|---------|---------|----------------|
| CER  | ~12% (8kHz)      | ~10%    | ~8%     | **< 8%**       |
| WER  | ~18%             | ~15%    | ~12%    | **< 12%**      |
| KWER (业务词) | ~15%    | ~12%    | ~10%    | **< 5%**       |

---

## 五、时间规划

| 阶段 | 任务 | 预计时间 | 累计时间 |
|------|------|----------|----------|
| Step 0 | 数据准备测试 | 10 分钟 | 10 分钟 |
| Stage 0 | 完整数据准备 | 4-6 小时 | 6 小时 |
| Stage 1 | Encoder 适配 | 2-3 天 | 3.25 天 |
| Stage 2 | Adapter 对齐 | 2-3 天 | 6.25 天 |
| Stage 3 | LoRA 业务适配 | 1-2 天 | 7.5 天 |
| 评估 | 结果分析 | 0.5 天 | **8 天** |

**总计**: ~8 天 (包含数据准备、训练、评估)

---

## 六、风险与应对

### 风险点 1: GPU OOM
- **应对**: 减小 batch_size (8192 → 4096)
- **配置**: 修改 conf/*.yaml 中的 `batch_size`

### 风险点 2: Stage 1 Loss 不收敛
- **应对**: 
  1. 降低学习率 (1e-4 → 5e-5)
  2. 增加 warmup_steps (1000 → 2000)
  3. 检查数据质量

### 风险点 3: Stage 3 过拟合 (数据少)
- **应对**: 
  1. 增大 gradient accumulation (2 → 4)
  2. 降低学习率 (1e-5 → 5e-6)
  3. Early stopping

---

## 七、Checkpoint 管理

### 自动保存
每个 Stage 完成后，最佳模型自动保存:
```
exp_output/exp/8k_telephone/
├── stage1_checkpoint.txt  → stage1_encoder_20260105_*/model.pt.avg
├── stage2_checkpoint.txt  → stage2_adapter_20260105_*/model.pt.avg
└── final_checkpoint.txt   → stage3_lora_20260105_*/model.pt.avg
```

### 手动备份
```bash
# 备份重要 checkpoint
cp -r exp_output/exp/8k_telephone/stage*_20260105_* /backup/fun_asr_nano_8k/
```

---

## 八、实验启动清单

- [x] 环境检查 (GPU, Python, FunASR)
- [x] 数据路径确认 (WenetSpeech, 电话数据)
- [x] 脚本准备 (run_experiment.sh, test_data_prep.sh)
- [x] 配置文件验证 (stage*.yaml)
- [x] 数据准备脚本增强 (filter_annotation_error)
- [ ] **运行小规模测试** (`bash test_data_prep.sh`)
- [ ] **启动 Stage 0** (`bash run_experiment.sh 0 0`)
- [ ] **启动 Stage 1** (`bash run_experiment.sh 1 1`)
- [ ] **启动 Stage 2** (`bash run_experiment.sh 2 2`)
- [ ] **启动 Stage 3** (`bash run_experiment.sh 3 3`)
- [ ] **运行评估** (`python evaluate_keywords.py`)

---

## 九、下一步行动

### 立即执行

```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# 1. 运行测试
bash test_data_prep.sh

# 2. 如果测试通过，启动完整实验
bash run_experiment.sh 0 3  # 一键运行所有阶段
```

### 监控方式

```bash
# 实时监控训练日志
watch -n 60 'tail -20 exp_output/exp/8k_telephone/*/train.log | grep -E "(loss|CER|epoch)"'

# 检查 GPU 使用
watch -n 5 'nvidia-smi'
```

---

**实验准备完成！准备开始执行。** 🚀
