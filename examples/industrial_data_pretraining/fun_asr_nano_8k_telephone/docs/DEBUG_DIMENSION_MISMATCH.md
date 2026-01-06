# 维度不匹配问题诊断与修复

## 🔍 问题诊断

### 症状
```
RuntimeError: The size of tensor a (17) must match the size of tensor b (18) 
at non-singleton dimension 2
```

### 根本原因链条

```
Warning: Drop Last Data
  ↓
Batch 样本长度分布不均 (17 vs 18 帧)
  ↓
位置编码缓冲大小固定
  ↓
某个批次超过预期长度
  ↓
位置编码维度不匹配
  ↓
RuntimeError 崩溃
```

### 详细分析

**1. Drop Last Data Warning 的作用**

```
Warning, 19th, b*t: 181*33=5973 > batch_size_sample_max: 2500, drop last data
```

- Batch 由 181 个样本组成
- 平均时间维度 T = 33 帧 (实际分布: 17-34 帧)
- 总token = 181 × 33 = 5973 > 2500 (限制)
- **动作**: 丢弃最后 N 个样本

**2. Drop 之后的问题**

丢弃数据后，剩余样本可能出现：
- Batch A: 样本 1-120，长度 17-18 帧 ✓
- Batch B: 样本 121-180，长度 **18-19 帧** ← 超出预期！

位置编码预期最大长度: 17 帧
实际输入长度: 18 或 19 帧
→ 维度不匹配!

**3. 为什么会有 17 vs 18 的差异？**

WavFrontend 计算流程（fs=16000）:
```
1. Fbank 特征提取
   - 音频时长: ~1.5s → 150 帧 (frame_shift=10ms)
   
2. LFR 下采样 (lfr_n=6)
   - 帧数 = (150 - 6) / 6 = 24 帧
   - 余数处理可能导致 ±1 的差异
   
3. CNN 子采样 (2层, stride=2)
   - 帧数 = 24 / 4 = 6 帧
   - 累积的舍入误差: 6±1 = 5-7 帧
   
4. 上采样到 16kHz 的影响
   - 8kHz → 16kHz 上采样增加了帧数
   - 某些音频多出 1-2 帧
```

最终结果: 样本长度在 15-20 帧之间波动
位置编码预期: 固定 17 帧
→ 某些样本长度 18，某些 17，无法对齐!

---

## ✅ 解决方案

### 推荐方案 1: 禁用 batch_size_sample_max 限制 (最简单)

**优点**: 一行改动，无需修改代码  
**缺点**: Batch 可能变大，显存占用增加

```yaml
# 编辑: conf/stage1_encoder_adapt.yaml
dataset_conf:
  batch_size_sample_max: 999999  # ← 改这里，禁用限制
  batch_type: token
  batch_size: 8192
```

**验证**:
```bash
# 修改后，应该看不到 "drop last data" 的警告
bash run_experiment.sh 1 1 2>&1 | grep "drop last data"
```

---

### 推荐方案 2: 同时调整多个参数 (更稳健)

```yaml
# conf/stage1_encoder_adapt.yaml
frontend_conf:
  fs: 8000  # ← 改为 8000，避免上采样引入的误差
  
dataset_conf:
  batch_size_sample_max: 999999    # ← 禁用
  max_speech_length: 12000         # ← 增加上限
  batch_size: 4096                 # ← 减小 batch 大小
```

**逻辑**:
- fs=8000: 8kHz 音频直接输入，无需上采样
- batch_size_sample_max: 不丢弃数据
- max_speech_length: 允许更长的序列
- batch_size: 减小以防显存溢出

---

### 方案 3: 修改位置编码最大长度 (需改代码)

如果前两个方案都不行，修改模型代码:

```python
# 文件: /workspace/share/LLMFunASR/funasr/models/sense_voice/model.py

# 找到这一行 (大约在第 996 行):
if "max_seq_len" not in kwargs:
    kwargs["max_seq_len"] = 512  # ← 改为更大的值

# 改为:
if "max_seq_len" not in kwargs:
    kwargs["max_seq_len"] = 2048  # 增加 4 倍
```

**说明**:
- max_seq_len = 512 表示位置编码支持最长 512 帧
- 但由于舍入误差，实际可用约 500-510 帧
- 增加至 2048 提供充分的缓冲

---

## 🚀 快速修复步骤

### Step 1: 修改配置 (1 分钟)

```bash
cd /workspace/share/LLMFunASR/examples/industrial_data_pretraining/fun_asr_nano_8k_telephone

# 备份原配置
cp conf/stage1_encoder_adapt.yaml conf/stage1_encoder_adapt.yaml.bak

# 修改配置
cat > /tmp/patch.yaml << 'PATCH'
--- a/conf/stage1_encoder_adapt.yaml
+++ b/conf/stage1_encoder_adapt.yaml
@@ -69,7 +69,7 @@
 # Dataset Configuration
 dataset_conf:
   index_ds: FunASR
   batch_sampler: BatchSampler
   batch_type: token
-  batch_size: 8192
+  batch_size_sample_max: 999999
PATCH

# 手动修改（用编辑器）
# 或使用 sed:
sed -i 's/batch_size_sample_max: [0-9]*/batch_size_sample_max: 999999/' conf/stage1_encoder_adapt.yaml
```

### Step 2: 验证修改

```bash
grep "batch_size_sample_max\|fs:" conf/stage1_encoder_adapt.yaml
```

应该看到:
```yaml
batch_size_sample_max: 999999
fs: 16000  # 或改为 8000
```

### Step 3: 重新启动训练

```bash
bash run_experiment.sh 1 1 2>&1 | tee stage1_run_v2.log

# 监控输出
tail -f stage1_run_v2.log | grep -E "Warning|Error|loss"
```

### Step 4: 验证修复

如果看到以下输出，说明修复成功:

```
✓ 没有 "drop last data" 警告
✓ 没有维度不匹配错误
✓ 开始输出 loss 值
```

---

## 🧪 测试修复

### 小规模测试

```bash
# 只用 100 个样本测试
python data/prepare_training_data.py validate \
    --input exp_output/data/simulated_8k_telephone/train_formatted.jsonl \
    --sample_size 100

# 检查样本长度分布
python << 'PYTHON'
import json
lengths = []
with open('exp_output/data/simulated_8k_telephone/train_formatted.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 1000: break
        data = json.loads(line)
        lengths.append(data['speech_length'])

print(f"Min: {min(lengths)}, Max: {max(lengths)}")
print(f"Mean: {sum(lengths)/len(lengths):.1f}")
print(f"Variance: {max(lengths) - min(lengths)}")
PYTHON
```

---

## 📊 对比分析

| 方案 | 修改复杂度 | 显存占用 | 训练速度 | 推荐度 |
|-----|----------|--------|--------|-------|
| 1: 禁用 limit | ⭐ 简单 | ↑ 增加 | ↓ 可能变慢 | ⭐⭐⭐⭐⭐ |
| 2: 调整多参数 | ⭐⭐ 中等 | → 不变 | → 不变 | ⭐⭐⭐⭐ |
| 3: 改代码 | ⭐⭐⭐ 复杂 | → 不变 | → 不变 | ⭐⭐⭐ |

**强烈推荐**: 方案 1 (禁用 batch_size_sample_max)

---

## ⚠️ 常见问题

**Q: 禁用 batch_size_sample_max 会导致 OOM 吗?**

A: 可能会。如果发生 OOM，改用方案 2:
```yaml
batch_size_sample_max: 5000   # 改为 5000 而不是 999999
```

**Q: 改 fs: 8000 有什么影响?**

A: 
- ✓ 避免上采样的舍入误差
- ✓ 更符合 8kHz 数据的原始特性
- ⚠ 但模型是在 16kHz 上预训练的，可能需要微调学习率

**Q: 应该同时改 frontend.fs 和 batch_size_sample_max 吗?**

A: 
- 如果显存充足: 只改 batch_size_sample_max
- 如果想最稳妥: 同时改两个

---

## 验证清单

- [ ] 备份原配置文件
- [ ] 修改 batch_size_sample_max 为 999999
- [ ] （可选）改 frontend.fs 为 8000
- [ ] 运行小规模测试
- [ ] 启动完整训练
- [ ] 监控前 5 个 batch 的 loss
- [ ] 确认没有维度错误

---

## 参考资源

- WavFrontend 代码: `/workspace/share/LLMFunASR/funasr/models/frontend/*.py`
- SenseVoice Encoder: `/workspace/share/LLMFunASR/funasr/models/sense_voice/model.py`
- 位置编码: `RelPositionalEncoding` 类

