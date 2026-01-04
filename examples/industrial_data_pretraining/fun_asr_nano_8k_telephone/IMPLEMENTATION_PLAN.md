# FunASR-Nano 8kHz 电话语音微调 - 实施方案

## 一、现状分析（基于事实）

### 1. FunASR-Nano 模型架构（来自 config.yaml）
```yaml
model: FunASRNano
audio_encoder: SenseVoiceEncoderSmall (冻结)
  - output_size: 512
  - 50 blocks transformer
  
llm: Qwen3-0.6B (冻结)
  - hidden_dim: 1024
  
audio_adaptor: Transformer (可训练)
  - downsample_rate: 1
  - encoder_dim: 512 → llm_dim: 1024
  - n_layer: 2
  
ctc_decoder: Transformer (可训练)
  - 用于CTC loss监督
  
frontend: WavFrontend
  - fs: 16000  # 关键：模型预训练在16kHz
  - n_mels: 80
  - lfr_m: 7, lfr_n: 6
```

### 2. 问题识别
- **问题1**: 当前 `config_8k_telephone.yaml` 配置错误
  - 使用了不存在的组件（WhisperFrontend with fs=8000）
  - audio_encoder 配置为 `"FunAudioLLM/Fun-ASR-Nano-2512"` 导致嵌套加载
  - llm 配置为 Qwen2.5-1.5B（应该用Qwen3-0.6B）

- **问题2**: 数据流不匹配
  - 8kHz 音频 → 需要上采样 → 16kHz → WavFrontend → 模型

## 二、混合方案实施（推荐 ✅）

### 核心原理
```
8kHz 电话音频（真实数据 or 模拟数据）
    ↓
[数据层：保留电话信道特征]
 - G.711 失真
 - 300-3400Hz 带通滤波
 - 环境噪声、串音
    ↓
[推理前：上采样到 16kHz]
 - 使用高质量重采样（保留特征）
    ↓
[WavFrontend: fs=16000]
 - 标准 16kHz 特征提取
    ↓
[SenseVoiceEncoder: 预训练权重]
 - 提取特征（冻结）
    ↓
[Audio Adaptor: 微调]
 - 适配电话域特征
    ↓
[Qwen3-0.6B + CTC Decoder]
 - 生成文本
```

### 方案优势
1. **不改变模型架构** - 复用预训练的16kHz前端和编码器
2. **数据层保留真实特征** - 电话失真在上采样前已存在
3. **训练高效** - 只微调 adaptor 和 ctc_decoder
4. **推理简单** - 8kHz输入自动上采样

## 三、实施步骤

### Step 1: 修复配置文件 `config_8k_telephone.yaml`

**关键修改点：**

```yaml
# 1. 使用正确的模型组件（与预训练模型一致）
model: FunASRNano

# 2. Audio Encoder - 从预训练模型加载
audio_encoder: "FunAudioLLM/Fun-ASR-Nano-2512"
audio_encoder_conf:
  hub: ms  # ModelScope
  freeze: true  # 冻结编码器
  freeze_layer_num: -1
  activation_checkpoint: false

# 3. LLM - 使用预训练的 Qwen3-0.6B
llm: Qwen3-0.6b
llm_conf:
  hub: hf  # 或 ms，取决于路径
  freeze: true
  llm_dtype: bf16
  init_param_path: Qwen3-0.6B  # 相对于 FunASR-Nano 模型目录
  use_lora: true  # 可选：使用 LoRA 微调
  lora_conf:
    freeze_lora: false
    task_type: CAUSAL_LM
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules:
      - q_proj
      - v_proj
    
# 4. Audio Adaptor - 可训练
audio_adaptor: Transformer
audio_adaptor_conf:
  downsample_rate: 1
  use_low_frame_rate: true
  ffn_dim: 2048
  llm_dim: 1024  # Qwen3-0.6B 的 hidden_size
  encoder_dim: 512  # SenseVoiceEncoder 输出
  n_layer: 2
  freeze: false  # 微调

# 5. CTC Decoder - 可训练
ctc_decoder: Transformer
detach_ctc_decoder: true
ctc_decoder_conf:
  downsample_rate: 1
  ffn_dim: 2048
  llm_dim: 512
  encoder_dim: 512
  n_layer: 5
  freeze: false  # 微调

# 6. Frontend - 保持 16kHz（关键！）
frontend: WavFrontend
frontend_conf:
  fs: 16000  # 必须是 16000
  window: hamming
  n_mels: 80
  frame_length: 25
  frame_shift: 10
  lfr_m: 7
  lfr_n: 6
  cmvn_file: null

# 7. 训练配置
train_conf:
  use_lora: ${llm_conf.use_lora}
  accum_grad: 1
  grad_clip: 5
  max_epoch: 20  # 可调整
  keep_nbest_models: 10
  log_interval: 10
  validate_interval: 500
  save_checkpoint_interval: 500
  avg_nbest_model: 5
  use_bf16: false
  use_deepspeed: false
  
# 8. 优化器 - 小学习率
optim: adamw
optim_conf:
  lr: 5.0e-05  # 5e-5
  weight_decay: 0.01

scheduler: warmuplr
scheduler_conf:
  warmup_steps: 500

# 9. 数据配置
dataset: FunASR
dataset_conf:
  index_ds: FunASR
  batch_sampler: BatchSampler
  batch_type: token
  batch_size: 4096  # 根据GPU调整
  max_token_length: 3000
  shuffle: true
  num_workers: 4
  audio_adaptor_downsample_rate: ${audio_adaptor_conf.downsample_rate}
  audio_encoder_downsample_rate: 6
  
  # CTC tokenizer
  ctc_tokenizer: SenseVoiceTokenizer
  ctc_target_normalize: true
  ctc_tokenizer_conf:
    vocab_path: null
    is_multilingual: true
    num_languages: 8749
    
  # Prompt配置
  prompt_classes: MultiContextPrompt
  prompt_conf:
    max_neg_hotwords_num: 0
    min_neg_hotwords_num: 0
    use_hist: false
    use_one_pass_result: true
    use_hotwords: true
    use_asr_hotwords: true

# 10. Tokenizer
tokenizer: HuggingfaceTokenizer
tokenizer_conf:
  init_param_path: ${llm_conf.init_param_path}

# 11. 初始化参数
init_param: "FunAudioLLM/Fun-ASR-Nano-2512/model.pt"
```

### Step 2: 修改训练脚本 - 添加音频上采样逻辑

**方案A: 在数据预处理时上采样（推荐）**

修改 `data/data_simulation.py`，在生成8kHz数据后，额外保存16kHz版本：

```python
def process_audio(self, audio, sr):
    """处理音频：降采样 → 电话效果 → 上采样回16kHz"""
    
    # 1. 降采样到8kHz（如果需要）
    if sr != self.target_fs:
        audio = self.resample(audio, sr, self.target_fs)
    
    # 2. 应用电话信道效果（300-3400Hz带通滤波）
    audio = self.apply_bandpass_filter(audio)
    
    # 3. G.711 codec 失真
    if self.add_codec:
        audio = self.apply_codec(audio)
    
    # 4. 添加噪声
    if self.add_noise:
        audio = self.add_telephone_noise(audio)
    
    # 5. 上采样回 16kHz（用于训练）
    audio_16k = self.resample(audio, self.target_fs, 16000)
    
    return audio_16k  # 返回16kHz，但保留了电话特征
```

**方案B: 在数据加载时上采样**

创建自定义 Dataset，在 `__getitem__` 中上采样：

```python
# 在 funasr/datasets/ 中添加
class TelephoneDataset:
    def __getitem__(self, idx):
        # 加载8kHz音频
        audio, sr = torchaudio.load(audio_path)
        
        # 上采样到16kHz
        if sr == 8000:
            resampler = torchaudio.transforms.Resample(8000, 16000)
            audio = resampler(audio)
        
        return audio, text
```

### Step 3: 训练脚本修改

`finetune_8k_telephone.sh` 关键修改：

```bash
# 配置文件指向修复后的版本
config="config_8k_telephone.yaml"

# 预训练模型初始化（关键！）
init_param="$MODELSCOPE_CACHE/models/FunAudioLLM/Fun-ASR-Nano-2512/model.pt"

# 训练命令
torchrun $DISTRIBUTED_ARGS \
  ${workspace}/../../../funasr/bin/train_ds.py \
  --config-path "${workspace}/conf" \
  --config-name "${config}" \
  ++init_param="${init_param}" \  # 从预训练模型初始化
  ++train_data_set_list="${train_data}" \
  ++valid_data_set_list="${val_data}" \
  ++output_dir="${output_dir}"
```

### Step 4: 数据准备建议

**选项1: 使用现有 8kHz 真实电话录音**
- 无需模拟，直接使用
- 在加载时上采样到 16kHz

**选项2: 从 16kHz 数据模拟**
```bash
# Stage -1: 数据模拟
bash finetune_8k_telephone.sh -1

# 模拟参数（已在脚本中）：
# - target_fs: 8000
# - low_freq: 300, high_freq: 3400
# - codec_type: mu-law
# - snr_db: 15-25
# - 输出：8kHz wav → 需要在训练前上采样
```

## 四、验证检查清单

### 训练前检查
- [ ] 配置文件中 `frontend_conf.fs = 16000`
- [ ] `init_param` 指向正确的预训练模型路径
- [ ] audio_encoder 和 llm 的 hub/init_param_path 正确
- [ ] 数据已经准备好（8kHz 或上采样后的 16kHz）
- [ ] GPU 内存足够（batch_size 需调整）

### 训练中监控
- [ ] 损失正常下降
- [ ] CTC loss 和 LM loss 都在优化
- [ ] 验证集 WER 逐步降低
- [ ] 没有 OOM 错误

### 训练后验证
- [ ] 用真实 8kHz 电话录音测试
- [ ] 对比不同场景（安静、嘈杂、口音）
- [ ] 与原始 16kHz 模型对比差异

## 五、预期效果

### 训练策略
```
冻结层：
  - Audio Encoder (SenseVoiceEncoder)：完全冻结
  - LLM (Qwen3-0.6B)：冻结 base，可选 LoRA

可训练层：
  - Audio Adaptor：全参数微调
  - CTC Decoder：全参数微调
  - LoRA（可选）：LLM 的低秩适配
```

### 性能预期
- **域适配**：电话失真特征 → Adaptor 学习补偿
- **保留能力**：预训练的语言理解能力不损失
- **收敛速度**：3-5 epochs 应该看到明显改进

## 六、故障排查

### 问题1: 加载预训练模型失败
```python
# 检查路径
init_param = "$MODELSCOPE_CACHE/models/FunAudioLLM/Fun-ASR-Nano-2512/model.pt"
ls -lh $init_param

# 检查 config.yaml 的组件名是否一致
```

### 问题2: Frontend 报错
```
Error: assert fs == 16000
```
**原因**: 配置中 `frontend_conf.fs` 设置为 8000  
**解决**: 必须设置为 16000，在加载数据时上采样

### 问题3: Shape mismatch
```
Error: encoder_dim mismatch
```
**原因**: audio_adaptor_conf 的 encoder_dim 不匹配  
**解决**: 必须是 512（SenseVoiceEncoder 输出）

## 七、下一步行动

1. **立即执行**:
   - [ ] 修复 `config_8k_telephone.yaml`
   - [ ] 修改 `data_simulation.py` 添加上采样
   - [ ] 更新 `finetune_8k_telephone.sh` 的 init_param

2. **测试验证**:
   - [ ] 用小数据集验证训练流程
   - [ ] 检查模型输出质量
   - [ ] 调整超参数

3. **生产部署**:
   - [ ] 全量数据训练
   - [ ] 模型评估和选择
   - [ ] 推理服务封装
