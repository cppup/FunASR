#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Prepare training data for Fun-ASR-Nano in correct JSONL format.

This script helps prepare train_data and val_data JSONL files in the format
required by Fun-ASR-Nano model.

Usage:
    python prepare_training_data.py --input train.csv --output train.jsonl
    python prepare_training_data.py --input val.csv --output val.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_audio_file(audio_path: str) -> bool:
    """Check if audio file exists."""
    return os.path.isfile(audio_path)


def create_jsonl_from_csv(
    csv_file: str,
    output_jsonl: str,
    source_col: str = "source",
    target_col: str = "target",
    make_absolute_path: bool = True,
    audio_dir: Optional[str] = None,
) -> int:
    """
    Convert CSV to JSONL format for Fun-ASR-Nano.
    
    Args:
        csv_file: Input CSV file path
        output_jsonl: Output JSONL file path
        source_col: Column name for audio source
        target_col: Column name for text target
        make_absolute_path: Convert relative paths to absolute
        audio_dir: Optional audio directory to prepend to relative paths
    
    Returns:
        Number of successfully processed lines
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Install with: pip install pandas")
        return 0
    
    logger.info(f"Reading CSV file: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to read CSV file: {e}")
        return 0
    
    # Check required columns
    if source_col not in df.columns or target_col not in df.columns:
        logger.error(f"CSV must contain '{source_col}' and '{target_col}' columns")
        logger.error(f"Available columns: {list(df.columns)}")
        return 0
    
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    success_count = 0
    skipped_count = 0
    
    logger.info(f"Converting {len(df)} records to JSONL format...")
    
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for idx, row in df.iterrows():
            try:
                source = str(row[source_col]).strip()
                target = str(row[target_col]).strip()
                
                # Skip empty entries
                if not source or not target:
                    skipped_count += 1
                    continue
                
                # Handle relative paths
                if make_absolute_path and not os.path.isabs(source):
                    if audio_dir:
                        source = os.path.join(audio_dir, source)
                    else:
                        source = os.path.abspath(source)
                
                # Validate audio file exists
                if not validate_audio_file(source):
                    logger.warning(f"Row {idx}: Audio file not found: {source}")
                    skipped_count += 1
                    continue
                
                # Build JSONL record
                record = {
                    "key": f"sample_{idx:06d}",
                    "source": source,
                    "target": target,
                }
                
                # Add optional fields if present
                if "source_len" in df.columns:
                    try:
                        record["source_len"] = int(row["source_len"])
                    except:
                        pass
                
                if "target_len" in df.columns:
                    try:
                        record["target_len"] = int(row["target_len"])
                    except:
                        pass
                
                if "domain" in df.columns:
                    record["domain"] = str(row["domain"]).strip()
                
                if "speaker_id" in df.columns:
                    record["speaker_id"] = str(row["speaker_id"]).strip()
                
                # Write to JSONL
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                success_count += 1
                
                if (success_count + skipped_count) % 100 == 0:
                    logger.info(f"Processed {success_count + skipped_count} records...")
                
            except Exception as e:
                logger.warning(f"Row {idx}: Error processing record: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"\nConversion complete:")
    logger.info(f"  Successfully converted: {success_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Output file: {output_jsonl}")
    
    return success_count


def create_jsonl_from_lists(
    audio_list: List[str],
    text_list: List[str],
    output_jsonl: str,
    make_absolute_path: bool = True,
    audio_dir: Optional[str] = None,
) -> int:
    """
    Create JSONL from separate audio and text lists.
    
    Args:
        audio_list: List of audio file paths
        text_list: List of transcription texts
        output_jsonl: Output JSONL file path
        make_absolute_path: Convert relative paths to absolute
        audio_dir: Optional audio directory
    
    Returns:
        Number of successfully processed lines
    """
    if len(audio_list) != len(text_list):
        logger.error(f"Audio list ({len(audio_list)}) and text list ({len(text_list)}) have different lengths")
        return 0
    
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    success_count = 0
    skipped_count = 0
    
    logger.info(f"Creating JSONL from {len(audio_list)} audio-text pairs...")
    
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for idx, (audio_path, text) in enumerate(zip(audio_list, text_list)):
            try:
                audio_path = str(audio_path).strip()
                text = str(text).strip()
                
                if not audio_path or not text:
                    skipped_count += 1
                    continue
                
                # Handle relative paths
                if make_absolute_path and not os.path.isabs(audio_path):
                    if audio_dir:
                        audio_path = os.path.join(audio_dir, audio_path)
                    else:
                        audio_path = os.path.abspath(audio_path)
                
                # Validate audio file exists
                if not validate_audio_file(audio_path):
                    logger.warning(f"Index {idx}: Audio file not found: {audio_path}")
                    skipped_count += 1
                    continue
                
                record = {
                    "key": f"sample_{idx:06d}",
                    "source": audio_path,
                    "target": text,
                }
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                success_count += 1
                
                if (success_count + skipped_count) % 100 == 0:
                    logger.info(f"Processed {success_count + skipped_count} pairs...")
                    
            except Exception as e:
                logger.warning(f"Index {idx}: Error processing pair: {e}")
                skipped_count += 1
                continue
    
    logger.info(f"\nConversion complete:")
    logger.info(f"  Successfully converted: {success_count}")
    logger.info(f"  Skipped: {skipped_count}")
    logger.info(f"  Output file: {output_jsonl}")
    
    return success_count


def create_jsonl_manual(
    output_jsonl: str,
    data: List[Dict],
) -> int:
    """
    Create JSONL from manually constructed data dictionaries.
    
    Args:
        output_jsonl: Output JSONL file path
        data: List of dictionaries with 'source' and 'target' keys
    
    Returns:
        Number of successfully written records
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    success_count = 0
    
    logger.info(f"Writing {len(data)} records to JSONL...")
    
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for idx, record in enumerate(data):
            try:
                if "source" not in record or "target" not in record:
                    logger.warning(f"Record {idx}: Missing 'source' or 'target' key")
                    continue
                
                if "key" not in record:
                    record["key"] = f"sample_{idx:06d}"
                
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                success_count += 1
                
            except Exception as e:
                logger.warning(f"Record {idx}: Error writing: {e}")
                continue
    
    logger.info(f"Successfully written: {success_count} records")
    logger.info(f"Output file: {output_jsonl}")
    
    return success_count


def validate_jsonl(jsonl_file: str, max_samples: int = 10) -> bool:
    """
    Validate JSONL file format.
    
    Args:
        jsonl_file: JSONL file path to validate
        max_samples: Max samples to check
    
    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Validating JSONL file: {jsonl_file}")
    
    if not os.path.isfile(jsonl_file):
        logger.error(f"File not found: {jsonl_file}")
        return False
    
    total_lines = 0
    valid_lines = 0
    
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                total_lines += 1
                
                try:
                    record = json.loads(line.strip())
                    
                    if "source" not in record or "target" not in record:
                        logger.warning(f"Line {idx}: Missing 'source' or 'target'")
                        continue
                    
                    source = record["source"]
                    if not validate_audio_file(source):
                        logger.warning(f"Line {idx}: Audio file not found: {source}")
                        continue
                    
                    valid_lines += 1
                    
                    if valid_lines <= max_samples:
                        logger.info(f"  Sample {valid_lines}: {record.get('key', f'line_{idx}')} - {record['target'][:50]}")
                    
                except json.JSONDecodeError:
                    logger.error(f"Line {idx}: Invalid JSON")
                    continue
                
                if idx >= 999 and idx % 1000 == 0:
                    logger.info(f"Checked {idx} lines...")
        
        logger.info(f"\nValidation result:")
        logger.info(f"  Total lines: {total_lines}")
        logger.info(f"  Valid lines: {valid_lines}")
        logger.info(f"  Invalid lines: {total_lines - valid_lines}")
        
        return valid_lines > 0
        
    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Fun-ASR-Nano in JSONL format"
    )
    
    parser.add_argument('--input', type=str, help='Input file (CSV, TXT, etc.)')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    parser.add_argument('--source_col', type=str, default='source',
                        help='Column name for audio source in CSV (default: source)')
    parser.add_argument('--target_col', type=str, default='target',
                        help='Column name for text target in CSV (default: target)')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Optional audio directory to prepend to relative paths')
    parser.add_argument('--absolute_path', action='store_true', default=True,
                        help='Convert relative paths to absolute (default: True)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate output JSONL file')
    
    args = parser.parse_args()
    
    # Create JSONL from CSV
    if args.input:
        if args.input.endswith('.csv'):
            success_count = create_jsonl_from_csv(
                args.input,
                args.output,
                source_col=args.source_col,
                target_col=args.target_col,
                make_absolute_path=args.absolute_path,
                audio_dir=args.audio_dir,
            )
        else:
            logger.error("Currently only CSV format is supported. Provide --input with .csv extension")
            sys.exit(1)
    else:
        logger.error("Please provide --input file")
        sys.exit(1)
    
    if success_count == 0:
        logger.error("No data was successfully processed")
        sys.exit(1)
    
    # Validate if requested
    if args.validate:
        is_valid = validate_jsonl(args.output)
        if not is_valid:
            logger.warning("JSONL validation found issues, but continuing...")


if __name__ == "__main__":
    main()
