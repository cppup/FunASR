# Fix train_data 和 val_data 的完整指南

## 目录

1. [问题诊断](#问题诊断)
2. [数据格式标准](#数据格式标准)
3. [修复步骤](#修复步骤)
4. [验证检查](#验证检查)
5. [常见错误](#常见错误)

---

## 问题诊断

如果训练失败或数据加载出错，通常是因为：

### 1. JSONL 格式不正确
```bash
# 检查 JSONL 文件有效性
python -c "
import json
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except:
            print(f'Line {i}: Invalid JSON')
"
```

### 2. 音频文件路径错误
```bash
# 检查音频文件是否存在
python -c "
import json
import os
with open('train.jsonl') as f:
    missing = 0
    for i, line in enumerate(f):
        if not line.strip():
            continue
        record = json.loads(line)
        if not os.path.isfile(record['source']):
            missing += 1
            if missing <= 5:
                print(f'Line {i}: Missing file {record[\"source\"]}')
    print(f'Total missing: {missing}')
"
```

### 3. 必填字段缺失
```bash
# 检查必填字段
python -c "
import json
required_fields = ['source', 'target']
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        record = json.loads(line)
        for field in required_fields:
            if field not in record:
                print(f'Line {i}: Missing field \"{field}\"')
"
```

---

## 数据格式标准

### 最小格式（必需）

```jsonl
{"source": "/path/to/audio.wav", "target": "转录文本"}
```

### 标准格式（推荐）

```jsonl
{
  "key": "sample_000001",
  "source": "/path/to/audio.wav",
  "target": "转录文本",
  "source_len": 10,
  "target_len": 5
}
```

### 完整格式（最佳）

```jsonl
{
  "key": "sample_000001",
  "source": "/path/to/audio.wav",
  "source_len": 10,
  "target": "转录文本",
  "target_len": 5,
  "domain": "telephone",
  "speaker_id": "speaker_001"
}
```

### 字段规范

| 字段 | 类型 | 必需 | 说明 | 示例 |
|------|------|------|------|------|
| source | string | ✓ | 音频文件路径 | "/data/audio/001.wav" |
| target | string | ✓ | 转录文本 | "这是转录内容" |
| key | string | × | 唯一标识符 | "sample_000001" |
| source_len | int | × | 音频时长(秒) | 10 |
| target_len | int | × | 文本长度(字) | 8 |
| domain | string | × | 数据领域 | "telephone" |
| speaker_id | string | × | 说话人ID | "speaker_001" |

---

## 修复步骤

### 方案 A: 从 CSV 转换（最简单）

如果已有 CSV 格式的数据：

```bash
# 1. 准备 CSV 文件，包含以下列：
#    - source: 音频文件路径
#    - target: 转录文本
#    其他列会被自动忽略

# 2. 使用转换脚本
python prepare_training_data.py \
    --input /path/to/train.csv \
    --output /path/to/train.jsonl \
    --audio_dir /path/to/audio \
    --validate
```

**CSV 示例：**
```
source,target,speaker_id
audio/001.wav,这是转录内容,speaker_1
audio/002.wav,第二段转录,speaker_2
```

### 方案 B: 从现有 JSONL 修复

如果 JSONL 格式有问题，使用修复脚本：

```python
import json
import os

def fix_jsonl(input_file, output_file, audio_dir=None):
    """修复 JSONL 文件"""
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                
                # 确保必填字段存在
                if 'source' not in record or 'target' not in record:
                    print(f"Line {idx}: 缺少必填字段，跳过")
                    continue
                
                # 修复路径
                source = record['source']
                if audio_dir and not os.path.isabs(source):
                    source = os.path.join(audio_dir, source)
                
                # 验证文件存在
                if not os.path.isfile(source):
                    print(f"Line {idx}: 文件不存在 {source}，跳过")
                    continue
                
                # 添加默认字段
                if 'key' not in record:
                    record['key'] = f'sample_{idx:06d}'
                
                # 清理文本
                record['target'] = str(record['target']).strip()
                record['source'] = source
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            except json.JSONDecodeError:
                print(f"Line {idx}: 无效JSON，跳过")
            except Exception as e:
                print(f"Line {idx}: 错误 {e}，跳过")

# 使用
fix_jsonl('train.jsonl', 'train_fixed.jsonl', audio_dir='/path/to/audio')
```

### 方案 C: 手动创建 JSONL

如果从零开始准备数据：

```python
import json
import os

# 步骤 1: 准备音频和文本列表
audio_files = [
    '/data/audio/001.wav',
    '/data/audio/002.wav',
    '/data/audio/003.wav',
]

transcriptions = [
    '这是第一段转录',
    '这是第二段转录',
    '这是第三段转录',
]

# 步骤 2: 创建 JSONL
output_file = 'train.jsonl'
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, (audio, text) in enumerate(zip(audio_files, transcriptions)):
        record = {
            'key': f'train_{idx:06d}',
            'source': os.path.abspath(audio),  # 使用绝对路径
            'target': text.strip(),
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f'Created {output_file} with {len(audio_files)} samples')
```

### 方案 D: 使用提供的准备脚本

```bash
# 1. 进入项目目录
cd /workspace/share/LLMFunASR

# 2. 运行准备脚本
python examples/prepare_data_example.py

# 3. 脚本会创建示例 JSONL 文件并显示用法说明
```

---

## 验证检查

### 检查清单

#### 1. 格式检查
```bash
# 检查 JSONL 有效性和样本数量
python -c "
import json
count = 0
with open('train.jsonl') as f:
    for line in f:
        if line.strip():
            try:
                json.loads(line)
                count += 1
            except:
                print('Invalid JSON line')
print(f'Valid records: {count}')
"
```

#### 2. 文件检查
```bash
# 检查所有音频文件是否存在
python -c "
import json, os
missing = 0
total = 0
with open('train.jsonl') as f:
    for line in f:
        if not line.strip(): continue
        record = json.loads(line)
        total += 1
        if not os.path.isfile(record['source']):
            missing += 1
            if missing <= 3:
                print(f'Missing: {record[\"source\"]}')
print(f'Missing files: {missing}/{total}')
"
```

#### 3. 内容检查
```bash
# 显示前 5 条样本
python -c "
import json
with open('train.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5: break
        record = json.loads(line)
        print(f'{i}: key={record.get(\"key\")}, target={record[\"target\"][:30]}...')
"
```

#### 4. 统计检查
```bash
# 统计数据分布
python -c "
import json
lengths = []
with open('train.jsonl') as f:
    for line in f:
        if line.strip():
            record = json.loads(line)
            text = record['target']
            lengths.append(len(text))
print(f'Total samples: {len(lengths)}')
print(f'Text length: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}')
"
```

### 完整验证脚本

```python
#!/usr/bin/env python3
import json
import os
import sys

def validate_jsonl(filename):
    """完整验证 JSONL 文件"""
    print(f"Validating {filename}...")
    print("=" * 50)
    
    valid_count = 0
    invalid_count = 0
    missing_files = 0
    
    text_lengths = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # 检查必填字段
                    if 'source' not in record or 'target' not in record:
                        print(f"Line {idx}: Missing required field")
                        invalid_count += 1
                        continue
                    
                    # 检查文件存在
                    source = record['source']
                    if not os.path.isfile(source):
                        missing_files += 1
                        if missing_files <= 3:
                            print(f"Line {idx}: Missing file - {source}")
                        continue
                    
                    # 收集统计信息
                    text = record['target']
                    text_lengths.append(len(text))
                    
                    valid_count += 1
                    
                    if valid_count <= 3:
                        print(f"Sample {valid_count}: {text[:50]}")
                
                except json.JSONDecodeError as e:
                    print(f"Line {idx}: Invalid JSON")
                    invalid_count += 1
        
        # 输出统计信息
        print("\n" + "=" * 50)
        print("Validation Results:")
        print(f"  Valid records: {valid_count}")
        print(f"  Invalid records: {invalid_count}")
        print(f"  Missing files: {missing_files}")
        print(f"  Total: {valid_count + invalid_count}")
        
        if text_lengths:
            print(f"\nText Statistics:")
            print(f"  Min length: {min(text_lengths)}")
            print(f"  Max length: {max(text_lengths)}")
            print(f"  Avg length: {sum(text_lengths)/len(text_lengths):.1f}")
        
        return valid_count > 0
    
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_jsonl.py <jsonl_file>")
        sys.exit(1)
    
    is_valid = validate_jsonl(sys.argv[1])
    sys.exit(0 if is_valid else 1)
```

---

## 常见错误

### 错误 1: 路径不是绝对路径

```
❌ 错误: {"source": "audio/001.wav", "target": "..."}
✓ 正确: {"source": "/data/audio/001.wav", "target": "..."}
```

**解决方案：**
```python
import os
record['source'] = os.path.abspath(record['source'])
```

### 错误 2: 音频文件不存在

```
❌ 错误: 指向不存在的文件 /path/to/missing.wav
✓ 正确: 音频文件存在且可读
```

**检查：**
```bash
ls -la /path/to/audio/*.wav | head -5
```

### 错误 3: 文本为空或包含特殊符号

```
❌ 错误: {"target": ""}
❌ 错误: {"target": "[笑] [停顿]"}
✓ 正确: {"target": "这是正常的转录文本"}
```

**清理文本：**
```python
text = text.strip()  # 移除前后空格
text = text.replace('[笑]', '').replace('[停顿]', '')  # 移除特殊标记
```

### 错误 4: JSON 格式不正确

```
❌ 错误: {"source": "/path/to/audio.wav", "target": "文本",}  # 尾部逗号
❌ 错误: {source: "/path/to/audio.wav"}  # 缺少引号
✓ 正确: {"source": "/path/to/audio.wav", "target": "文本"}
```

### 错误 5: 字符编码问题

```
❌ 错误: 使用 GB2312 或其他编码
✓ 正确: 必须使用 UTF-8 编码
```

**检查编码：**
```bash
file -i train.jsonl  # 应显示 charset=utf-8
```

---

## 最佳实践

### 1. 使用绝对路径

```python
import os
source = os.path.abspath(record['source'])
```

### 2. 验证数据完整性

```python
import os
assert os.path.isfile(record['source']), f"Missing file: {record['source']}"
assert record['target'].strip(), "Empty target text"
```

### 3. 标准化文本

```python
# 移除前后空格
text = record['target'].strip()

# 移除多余空格
text = ' '.join(text.split())

# 可选：转小写（仅英文）
# text = text.lower()
```

### 4. 添加唯一标识符

```python
if 'key' not in record:
    record['key'] = f"sample_{idx:06d}"
```

### 5. 分离训练/验证集

```bash
# 按 8:2 比例分割
split_index=$(($(wc -l < data.jsonl) * 80 / 100))
head -n $split_index data.jsonl > train.jsonl
tail -n +$((split_index + 1)) data.jsonl > val.jsonl
```

---

## 快速修复脚本

将以下脚本保存为 `fix_data.py`：

```python
#!/usr/bin/env python3
import json
import os
import sys

def fix_data(input_file, output_file):
    """快速修复 JSONL 文件"""
    fixed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
                
                # 检查必填字段
                if 'source' not in record or 'target' not in record:
                    skipped_count += 1
                    continue
                
                # 转换为绝对路径
                record['source'] = os.path.abspath(record['source'])
                
                # 验证文件存在
                if not os.path.isfile(record['source']):
                    skipped_count += 1
                    continue
                
                # 清理文本
                record['target'] = str(record['target']).strip()
                
                # 添加 key
                if 'key' not in record:
                    record['key'] = f'sample_{idx:06d}'
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                fixed_count += 1
            except Exception as e:
                skipped_count += 1
    
    print(f"Fixed: {fixed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fix_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    fix_data(sys.argv[1], sys.argv[2])
```

**使用：**
```bash
python fix_data.py train_broken.jsonl train.jsonl
python fix_data.py val_broken.jsonl val.jsonl
```

---

## 总结

修复 train_data 和 val_data 的关键步骤：

1. ✓ **格式检查**：确保是有效的 JSONL
2. ✓ **必填字段**：`source` 和 `target` 必须存在
3. ✓ **路径检查**：使用绝对路径，验证文件存在
4. ✓ **文本清理**：移除特殊符号，标准化编码
5. ✓ **验证数据**：运行检查脚本，确保数据有效

使用提供的脚本可以快速完成这些步骤。
