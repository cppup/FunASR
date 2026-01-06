#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Quick fix script for train_data and val_data JSONL files.

This script repairs common issues in JSONL files to make them compatible with Fun-ASR-Nano:
- Fixes relative paths to absolute paths
- Validates audio file existence
- Cleans up text content
- Adds missing key field
- Removes invalid records

Usage:
    python fix_training_data.py train.jsonl train_fixed.jsonl
    python fix_training_data.py val.jsonl val_fixed.jsonl
"""

import json
import os
import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_jsonl_file(
    input_file: str,
    output_file: str,
    audio_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    Fix JSONL file for Fun-ASR-Nano compatibility.
    
    Args:
        input_file: Input JSONL file path
        output_file: Output JSONL file path
        audio_dir: Optional base directory for audio files
        verbose: Print detailed information
    
    Returns:
        Dictionary with statistics
    """
    
    stats = {
        'total': 0,
        'fixed': 0,
        'invalid_json': 0,
        'missing_required_fields': 0,
        'missing_audio_file': 0,
        'empty_text': 0,
        'other_errors': 0,
    }
    
    logger.info(f"Reading input file: {input_file}")
    
    if not os.path.isfile(input_file):
        logger.error(f"Input file not found: {input_file}")
        return stats
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for idx, line in enumerate(fin):
            # Skip empty lines
            if not line.strip():
                continue
            
            stats['total'] += 1
            
            try:
                # Parse JSON
                record = json.loads(line)
            except json.JSONDecodeError as e:
                if verbose:
                    logger.warning(f"Line {idx}: Invalid JSON - {e}")
                stats['invalid_json'] += 1
                continue
            
            try:
                # Check required fields
                if 'source' not in record:
                    logger.warning(f"Line {idx}: Missing 'source' field")
                    stats['missing_required_fields'] += 1
                    continue
                
                if 'target' not in record:
                    logger.warning(f"Line {idx}: Missing 'target' field")
                    stats['missing_required_fields'] += 1
                    continue
                
                # Fix source path
                source = str(record['source']).strip()
                if not source:
                    logger.warning(f"Line {idx}: Empty 'source' field")
                    stats['missing_audio_file'] += 1
                    continue
                
                # Handle relative paths
                if not os.path.isabs(source):
                    if audio_dir:
                        source = os.path.join(audio_dir, source)
                    source = os.path.abspath(source)
                
                # Validate file exists
                if not os.path.isfile(source):
                    logger.warning(f"Line {idx}: Audio file not found - {source}")
                    stats['missing_audio_file'] += 1
                    continue
                
                # Fix target text
                target = str(record['target']).strip()
                if not target:
                    logger.warning(f"Line {idx}: Empty 'target' field")
                    stats['empty_text'] += 1
                    continue
                
                # Clean target text
                # Remove extra whitespace
                target = ' '.join(target.split())
                
                # Add missing key field
                if 'key' not in record:
                    record['key'] = f"sample_{idx:06d}"
                
                # Update fields
                record['source'] = source
                record['target'] = target
                
                # Write fixed record
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['fixed'] += 1
                
                if verbose and stats['fixed'] <= 3:
                    logger.info(f"Sample {stats['fixed']}: {target[:50]}...")
                
            except Exception as e:
                logger.warning(f"Line {idx}: Error processing - {e}")
                stats['other_errors'] += 1
                continue
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Fix JSONL files for Fun-ASR-Nano training"
    )
    
    parser.add_argument('input', type=str, help='Input JSONL file')
    parser.add_argument('output', type=str, help='Output JSONL file')
    parser.add_argument('--audio_dir', type=str, default=None,
                        help='Base directory for audio files (if relative paths)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed information')
    
    args = parser.parse_args()
    
    logger.info("Starting JSONL file repair...")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {args.output}")
    if args.audio_dir:
        logger.info(f"  Audio dir: {args.audio_dir}")
    
    # Fix the file
    stats = fix_jsonl_file(
        args.input,
        args.output,
        audio_dir=args.audio_dir,
        verbose=args.verbose,
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("REPAIR STATISTICS")
    print("=" * 60)
    print(f"Total records:              {stats['total']}")
    print(f"Successfully fixed:         {stats['fixed']}")
    print(f"  Invalid JSON:             {stats['invalid_json']}")
    print(f"  Missing required fields:  {stats['missing_required_fields']}")
    print(f"  Missing audio files:      {stats['missing_audio_file']}")
    print(f"  Empty text:               {stats['empty_text']}")
    print(f"  Other errors:             {stats['other_errors']}")
    print("=" * 60)
    
    if stats['fixed'] > 0:
        success_rate = stats['fixed'] / stats['total'] * 100
        print(f"Success rate: {success_rate:.1f}%")
        print(f"\n✓ Fixed file saved to: {args.output}")
        return 0
    else:
        logger.error("No records were successfully processed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
