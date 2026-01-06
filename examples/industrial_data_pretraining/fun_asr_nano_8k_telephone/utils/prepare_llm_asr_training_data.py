#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Data preparation tool for Fun-ASR-Nano 8kHz telephone channel fine-tuning.

Features:
1. Convert simple format to Fun-ASR-Nano messages format
2. Upsample 8kHz audio to 16kHz (for WavFrontend compatibility)
3. Validate data format
4. Calculate speech_length from audio files

Input format (simple):
    {"source": "/path/to/audio.wav", "target": "transcription text"}

Output format (Fun-ASR-Nano):
    {
        "key": "sample_001",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "语音转写：<|startofspeech|>!/path/to/audio.wav<|endofspeech|>"},
            {"role": "assistant", "content": "transcription text"}
        ],
        "speech_length": 800,
        "text_length": 20
    }
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def calculate_speech_length(audio_path, target_fs=16000, frame_shift=10, lfr_n=6):
    """
    Calculate speech_length for Fun-ASR-Nano training.
    
    Args:
        audio_path: Path to audio file
        target_fs: Target sampling rate (default: 16000)
        frame_shift: Frame shift in ms (default: 10)
        lfr_n: LFR factor (default: 6)
    
    Returns:
        Tuple of (speech_length, duration_sec, sample_rate)
    """
    try:
        info = sf.info(audio_path)
        duration_sec = info.duration
        sample_rate = info.samplerate
        
        # Calculate frames based on target_fs
        # Even if audio is 8kHz, we assume it will be upsampled to 16kHz
        num_frames = int(duration_sec * 1000 / frame_shift)  # frames at 100 fps
        speech_length = num_frames // lfr_n  # LFR downsampling
        
        return speech_length, duration_sec, sample_rate
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return None, None, None


def upsample_audio(audio, orig_fs, target_fs=16000):
    """
    Upsample audio to target sampling rate.
    
    Args:
        audio: Audio signal
        orig_fs: Original sampling rate
        target_fs: Target sampling rate (default: 16000)
    
    Returns:
        Upsampled audio at target_fs
    """
    if orig_fs == target_fs:
        return audio
    
    num_samples = int(len(audio) * target_fs / orig_fs)
    resampled = signal.resample(audio, num_samples)
    return resampled


def process_single_sample(args):
    """
    Process a single sample (worker function for parallel processing).
    
    Args:
        args: Tuple of (line_no, data, output_audio_dir, target_fs, do_upsample, task_template, filter_annotation_error)
    
    Returns:
        Tuple of (line_no, output_data, success, error)
    """
    line_no, data, output_audio_dir, target_fs, do_upsample, task_template, filter_annotation_error = args
    
    try:
        audio_path = data.get('source', data.get('audio', ''))
        target_text = data.get('target', data.get('text', ''))
        
        if not audio_path:
            return (line_no, None, False, "Missing 'source' or 'audio' field")
        
        if not target_text:
            return (line_no, None, False, "Missing 'target' or 'text' field")
        
        # Filter annotation errors
        if filter_annotation_error and '<ANNOTATION_ERROR>' in target_text:
            return (line_no, None, False, "Annotation error detected")
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            return (line_no, None, False, f"Audio file not found: {audio_path}")
        
        # Calculate speech_length
        speech_length, duration_sec, sample_rate = calculate_speech_length(audio_path, target_fs)
        
        if speech_length is None:
            return (line_no, None, False, f"Failed to read audio: {audio_path}")
        
        # Upsample if needed
        final_audio_path = audio_path
        if do_upsample and sample_rate != target_fs:
            try:
                audio, orig_fs = sf.read(audio_path)
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Upsample
                audio_resampled = upsample_audio(audio, orig_fs, target_fs)
                
                # Save upsampled audio
                basename = os.path.basename(audio_path)
                name, ext = os.path.splitext(basename)
                output_filename = f"{name}_{target_fs // 1000}k{ext}"
                output_path = os.path.join(output_audio_dir, output_filename)
                
                os.makedirs(output_audio_dir, exist_ok=True)
                sf.write(output_path, audio_resampled, target_fs)
                
                final_audio_path = output_path
            except Exception as e:
                return (line_no, None, False, f"Failed to upsample audio: {e}")
        
        # Build output data
        key = data.get('key', f"sample_{line_no:08d}")
        
        # Build user content with task template
        user_content = f"{task_template}<|startofspeech|>!{final_audio_path}<|endofspeech|>"
        
        output = {
            "key": key,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": target_text}
            ],
            "speech_length": speech_length,
            "text_length": len(target_text)
        }
        
        # Copy optional fields
        if 'hotwords' in data:
            output['hotwords'] = data['hotwords']
        if 'asr_hotwords' in data:
            output['asr_hotwords'] = data['asr_hotwords']
        if 'language' in data:
            output['language'] = data['language']
        
        return (line_no, output, True, None)
    
    except Exception as e:
        return (line_no, None, False, str(e))


def convert_to_funasr_format(
    input_jsonl,
    output_jsonl,
    output_audio_dir=None,
    target_fs=16000,
    do_upsample=False,
    task_template="语音转写：",
    num_workers=1,
    min_speech_length=10,
    max_speech_length=8000,
    min_text_length=1,
    max_text_length=2048,
    filter_annotation_error=False,
):
    """
    Convert simple JSONL format to Fun-ASR-Nano training format.
    
    Args:
        input_jsonl: Input JSONL file path
        output_jsonl: Output JSONL file path
        output_audio_dir: Output directory for upsampled audio (required if do_upsample=True)
        target_fs: Target sampling rate (default: 16000)
        do_upsample: Whether to upsample audio (default: False)
        task_template: Task template for user content (default: "语音转写：")
        num_workers: Number of parallel workers (default: 1)
        min_speech_length: Minimum speech_length filter (default: 10)
        max_speech_length: Maximum speech_length filter (default: 8000)
        min_text_length: Minimum text_length filter (default: 1)
        max_text_length: Maximum text_length filter (default: 2048)
        filter_annotation_error: Filter out samples with <ANNOTATION_ERROR> tag (default: False)
    """
    if do_upsample and output_audio_dir is None:
        raise ValueError("output_audio_dir is required when do_upsample=True")
    
    if output_audio_dir:
        os.makedirs(output_audio_dir, exist_ok=True)
    
    os.makedirs(os.path.dirname(output_jsonl) or '.', exist_ok=True)
    
    # Read input data
    samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                samples.append((line_no, data, output_audio_dir, target_fs, do_upsample, task_template, filter_annotation_error))
            except json.JSONDecodeError as e:
                print(f"Line {line_no}: JSON decode error: {e}")
    
    print(f"Processing {len(samples)} samples...")
    
    results = []
    
    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_sample, args): args[0] for args in samples}
            
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if completed % max(1, len(samples) // 10) == 0:
                    print(f"Processed {completed}/{len(samples)} samples...")
    else:
        # Serial processing
        for args in samples:
            results.append(process_single_sample(args))
            if len(results) % 100 == 0:
                print(f"Processed {len(results)}/{len(samples)} samples...")
    
    # Sort by line number
    results.sort(key=lambda x: x[0])
    
    # Filter and write output
    success_count = 0
    filtered_count = 0
    error_count = 0
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for line_no, output, success, error in results:
            if not success:
                print(f"Line {line_no}: Error - {error}")
                error_count += 1
                continue
            
            # Apply filters
            speech_length = output['speech_length']
            text_length = output['text_length']
            
            if speech_length < min_speech_length:
                filtered_count += 1
                continue
            if speech_length > max_speech_length:
                filtered_count += 1
                continue
            if text_length < min_text_length:
                filtered_count += 1
                continue
            if text_length > max_text_length:
                filtered_count += 1
                continue
            
            f.write(json.dumps(output, ensure_ascii=False) + '\n')
            success_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary")
    print(f"{'='*60}")
    print(f"Input file: {input_jsonl}")
    print(f"Output file: {output_jsonl}")
    print(f"Total samples: {len(samples)}")
    print(f"Successful: {success_count}")
    print(f"Filtered: {filtered_count}")
    print(f"Errors: {error_count}")
    print(f"{'='*60}\n")


def validate_funasr_format(jsonl_path, check_audio=False, sample_size=None):
    """
    Validate Fun-ASR-Nano training data format.
    
    Args:
        jsonl_path: Path to JSONL file
        check_audio: Whether to check if audio files exist (default: False)
        sample_size: Maximum number of samples to validate (default: None = all)
    
    Returns:
        True if valid, False otherwise
    """
    import re
    
    errors = []
    warnings = []
    valid_count = 0
    total_speech_length = 0
    total_text_length = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            # Skip if sample_size is set and we've reached the limit
            if sample_size is not None and line_no > sample_size:
                break
            
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_no}: JSON decode error - {e}")
                continue
            
            # Check required fields
            if 'messages' not in data:
                errors.append(f"Line {line_no}: Missing 'messages' field")
                continue
            
            if 'speech_length' not in data:
                warnings.append(f"Line {line_no}: Missing 'speech_length' field")
            else:
                total_speech_length += data['speech_length']
            
            if 'text_length' not in data:
                warnings.append(f"Line {line_no}: Missing 'text_length' field")
            else:
                total_text_length += data['text_length']
            
            # Check messages structure
            messages = data['messages']
            has_user = False
            has_assistant = False
            
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    errors.append(f"Line {line_no}: Message missing 'role' or 'content'")
                    continue
                
                if msg['role'] == 'user':
                    has_user = True
                    content = msg['content']
                    
                    # Check audio path format
                    if '<|startofspeech|>' in content:
                        match = re.search(r'<\|startofspeech\|>!(.+?)<\|endofspeech\|>', content)
                        if not match:
                            errors.append(f"Line {line_no}: Invalid audio path format")
                        elif check_audio:
                            audio_path = match.group(1)
                            if not os.path.exists(audio_path):
                                errors.append(f"Line {line_no}: Audio file not found - {audio_path}")
                
                if msg['role'] == 'assistant':
                    has_assistant = True
            
            if not has_user:
                errors.append(f"Line {line_no}: Missing 'user' role message")
            if not has_assistant:
                errors.append(f"Line {line_no}: Missing 'assistant' role message")
            
            # Count valid if no errors for this line
            line_errors = [e for e in errors if e.startswith(f"Line {line_no}:")]
            if not line_errors:
                valid_count += 1
    
    # Calculate statistics
    avg_speech_length = total_speech_length / valid_count if valid_count > 0 else 0
    avg_text_length = total_text_length / valid_count if valid_count > 0 else 0
    
    # Print report
    print(f"\n{'='*60}")
    print(f"Validation Report: {jsonl_path}")
    print(f"{'='*60}")
    print(f"Valid samples: {valid_count}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    print(f"Avg speech_length: {avg_speech_length:.1f}")
    print(f"Avg text_length: {avg_text_length:.1f}")
    
    if errors:
        print(f"\nErrors (first 10):")
        for err in errors[:10]:
            print(f"  ❌ {err}")
    
    if warnings:
        print(f"\nWarnings (first 10):")
        for warn in warnings[:10]:
            print(f"  ⚠️ {warn}")
    
    print(f"{'='*60}\n")
    
    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Data preparation tool for Fun-ASR-Nano 8kHz telephone fine-tuning"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert simple format to Fun-ASR-Nano format')
    convert_parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    convert_parser.add_argument('--output', type=str, required=True, help='Output JSONL file')
    convert_parser.add_argument('--output_audio_dir', type=str, default=None, help='Output directory for upsampled audio')
    convert_parser.add_argument('--target_fs', type=int, default=16000, help='Target sampling rate (default: 16000)')
    convert_parser.add_argument('--do_upsample', action='store_true', help='Upsample audio to target_fs')
    convert_parser.add_argument('--task_template', type=str, default='语音转写：', help='Task template (default: 语音转写：)')
    convert_parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    convert_parser.add_argument('--min_speech_length', type=int, default=10, help='Min speech_length filter')
    convert_parser.add_argument('--max_speech_length', type=int, default=8000, help='Max speech_length filter')
    convert_parser.add_argument('--min_text_length', type=int, default=1, help='Min text_length filter')
    convert_parser.add_argument('--max_text_length', type=int, default=2048, help='Max text_length filter')
    convert_parser.add_argument('--filter_annotation_error', action='store_true', 
                                help='Filter out samples with <ANNOTATION_ERROR>')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate Fun-ASR-Nano data format')
    validate_parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    validate_parser.add_argument('--check_audio', action='store_true', help='Check if audio files exist')
    validate_parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to validate')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        convert_to_funasr_format(
            input_jsonl=args.input,
            output_jsonl=args.output,
            output_audio_dir=args.output_audio_dir,
            target_fs=args.target_fs,
            do_upsample=args.do_upsample,
            task_template=args.task_template,
            num_workers=args.num_workers,
            min_speech_length=args.min_speech_length,
            max_speech_length=args.max_speech_length,
            min_text_length=args.min_text_length,
            max_text_length=args.max_text_length,
            filter_annotation_error=args.filter_annotation_error,
        )
    
    elif args.command == 'validate':
        success = validate_funasr_format(args.input, args.check_audio, args.sample_size)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
