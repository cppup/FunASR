#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Example: Prepare sample training data for Fun-ASR-Nano.

This script demonstrates how to prepare training and validation data
in the correct JSONL format required by Fun-ASR-Nano model.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_jsonl(
    output_path: str,
    data_records: List[Dict],
) -> None:
    """Create a sample JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in data_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    logger.info(f"Created: {output_path}")


def main():
    """
    Prepare sample training and validation data.
    
    Prerequisites:
    - Audio files should be in WAV format, 16kHz, mono
    - Text transcriptions should be clean Chinese or English
    """
    
    # Step 1: Define your audio and transcription data
    # Format: {"source": "/path/to/audio.wav", "target": "transcription"}
    
    train_data = [
        {
            "key": "train_001",
            "source": "/data/audio/train_001.wav",
            "target": "机器学习是人工智能的一个重要分支",
            "domain": "technical",
        },
        {
            "key": "train_002",
            "source": "/data/audio/train_002.wav",
            "target": "深度神经网络在图像识别中取得了突破性进展",
            "domain": "technical",
        },
        {
            "key": "train_003",
            "source": "/data/audio/train_003.wav",
            "target": "自然语言处理是计算机科学的一个重要领域",
            "domain": "technical",
        },
        {
            "key": "train_004",
            "source": "/data/audio/train_004.wav",
            "target": "声音识别技术在智能语音助手中得到应用",
            "domain": "technical",
        },
        {
            "key": "train_005",
            "source": "/data/audio/train_005.wav",
            "target": "我们正在开发一个新的语音识别系统",
            "domain": "business",
        },
    ]
    
    val_data = [
        {
            "key": "val_001",
            "source": "/data/audio/val_001.wav",
            "target": "人工智能的发展推动了社会的进步",
            "domain": "technical",
        },
        {
            "key": "val_002",
            "source": "/data/audio/val_002.wav",
            "target": "基础设施建设是经济发展的重要基础",
            "domain": "business",
        },
    ]
    
    # Step 2: Create output directory
    output_dir = "/workspace/data/funasr"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 3: Write JSONL files
    train_jsonl = os.path.join(output_dir, "train.jsonl")
    val_jsonl = os.path.join(output_dir, "val.jsonl")
    
    create_sample_jsonl(train_jsonl, train_data)
    create_sample_jsonl(val_jsonl, val_data)
    
    # Step 4: Validate format
    logger.info("\n=== Data Format Summary ===")
    logger.info(f"Training data: {train_jsonl}")
    logger.info(f"Validation data: {val_jsonl}")
    logger.info(f"Training samples: {len(train_data)}")
    logger.info(f"Validation samples: {len(val_data)}")
    
    logger.info("\n=== Sample Records ===")
    logger.info("Training sample:")
    logger.info(json.dumps(train_data[0], ensure_ascii=False, indent=2))
    logger.info("\nValidation sample:")
    logger.info(json.dumps(val_data[0], ensure_ascii=False, indent=2))
    
    # Step 5: Instructions for using the data
    print("""
=================================================================
           Data Preparation Complete!
=================================================================

Next steps:

1. Verify audio files exist at the paths specified in JSONL:
   find /data/audio -name "*.wav" | wc -l

2. For 8kHz telephone scenario simulation:
   python examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/data/data_simulation.py \\
       --input {}/train.jsonl \\
       --output {}/train_8k.jsonl \\
       --output_audio_dir {}/audio_8k \\
       --num_workers 64

3. For training, update the config and run:
   bash examples/industrial_data_pretraining/fun_asr_nano_8k_telephone/finetune_8k_telephone.sh

4. Model and pretrained weights location:
   $MODELSCOPE_CACHE/models/
   
   Set cache directory:
   export MODELSCOPE_CACHE=/path/to/cache

=================================================================
    JSONL Format Requirements
=================================================================

Required fields:
  - source: Path to WAV file (absolute path recommended)
  - target: Transcription text (clean, no special characters)

Optional fields:
  - key: Unique identifier for the sample
  - source_len: Audio length in seconds (for reference)
  - target_len: Text length in characters (for reference)
  - domain: Domain tag (e.g., "technical", "business")
  - speaker_id: Speaker identifier (for multi-speaker training)

Audio format:
  - Sampling rate: 16000 Hz (mono)
  - Format: WAV, MP3, FLAC, etc. (librosa supported)
  - Duration: Recommended 5-30 seconds per sample

Text format:
  - Encoding: UTF-8
  - Language: Chinese, English, or mixed
  - No special symbols: [笑], [停顿], etc.

=================================================================
""".format(output_dir, output_dir, output_dir))


if __name__ == "__main__":
    main()
