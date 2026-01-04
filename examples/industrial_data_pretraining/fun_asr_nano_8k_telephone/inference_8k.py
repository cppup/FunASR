#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Inference script for Fun-ASR-Nano 8kHz telephone channel model.
Supports single file, batch, and JSONL evaluation with CER/WER metrics.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from funasr import AutoModel


def compute_cer(ref, hyp):
    """
    Compute Character Error Rate (CER).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
    
    Returns:
        CER value (0-1)
    """
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    # Simple Levenshtein distance calculation
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(ref)][len(hyp)] / len(ref)


def compute_wer(ref, hyp):
    """
    Compute Word Error Rate (WER).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
    
    Returns:
        WER value (0-1)
    """
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    # Simple Levenshtein distance calculation for words
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
    
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def inference_single_file(model, audio_path, hotwords=None):
    """
    Perform inference on a single audio file.
    
    Args:
        model: AutoModel instance
        audio_path: Path to audio file
        hotwords: Optional hotwords for biased decoding
    
    Returns:
        Recognition result
    """
    try:
        res = model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            hotwords=hotwords,
        )
        return res[0] if res else None
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def inference_batch(model, audio_paths, batch_size=1, hotwords=None):
    """
    Perform batch inference on multiple audio files.
    
    Args:
        model: AutoModel instance
        audio_paths: List of audio file paths
        batch_size: Batch size for inference
        hotwords: Optional hotwords for biased decoding
    
    Returns:
        List of recognition results
    """
    results = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch = audio_paths[i:i+batch_size]
        
        try:
            res = model.generate(
                input=batch,
                cache={},
                batch_size=len(batch),
                hotwords=hotwords,
            )
            results.extend(res)
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {e}")
            results.extend([None] * len(batch))
    
    return results


def inference_jsonl(
    model, 
    jsonl_path, 
    output_dir, 
    batch_size=1, 
    hotwords=None,
    audio_key="source",
    text_key="target",
):
    """
    Perform inference on JSONL dataset and compute metrics.
    
    Args:
        model: AutoModel instance
        jsonl_path: Path to JSONL file
        output_dir: Output directory for results
        batch_size: Batch size for inference
        hotwords: Optional hotwords for biased decoding
        audio_key: Key in JSONL for audio file path
        text_key: Key in JSONL for reference text
    
    Returns:
        Dictionary with metrics and results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSONL
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]
    
    print(f"Loaded {len(data)} samples from {jsonl_path}")
    
    # Prepare batches
    audio_paths = []
    references = []
    sample_ids = []
    
    for i, item in enumerate(data):
        if audio_key in item:
            audio_paths.append(item[audio_key])
            references.append(item.get(text_key, ""))
            sample_ids.append(item.get("id", f"sample_{i}"))
        else:
            print(f"Warning: Missing key '{audio_key}' in sample {i}")
    
    print(f"Processing {len(audio_paths)} audio files...")
    
    # Perform inference
    start_time = time.time()
    results = inference_batch(model, audio_paths, batch_size=batch_size, hotwords=hotwords)
    elapsed_time = time.time() - start_time
    
    # Compute metrics
    total_cer = 0.0
    total_wer = 0.0
    valid_count = 0
    
    output_results = []
    
    for i, (result, ref) in enumerate(zip(results, references)):
        if result is None:
            print(f"Warning: No result for sample {sample_ids[i]}")
            continue
        
        hyp = result.get("text", "")
        
        # Compute CER and WER
        cer = compute_cer(ref, hyp) if ref else 0.0
        wer = compute_wer(ref, hyp) if ref else 0.0
        
        total_cer += cer
        total_wer += wer
        valid_count += 1
        
        output_results.append({
            "id": sample_ids[i],
            "audio": audio_paths[i],
            "reference": ref,
            "hypothesis": hyp,
            "cer": cer,
            "wer": wer,
        })
    
    # Compute average metrics
    avg_cer = total_cer / valid_count if valid_count > 0 else 0.0
    avg_wer = total_wer / valid_count if valid_count > 0 else 0.0
    rtf = elapsed_time / valid_count if valid_count > 0 else 0.0
    
    # Save results
    results_file = os.path.join(output_dir, "results.jsonl")
    with open(results_file, 'w', encoding='utf-8') as f:
        for item in output_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save metrics
    metrics = {
        "total_samples": len(audio_paths),
        "valid_samples": valid_count,
        "failed_samples": len(audio_paths) - valid_count,
        "cer": avg_cer,
        "wer": avg_wer,
        "rtf": rtf,
        "total_time": elapsed_time,
    }
    
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Inference Results Summary")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"Failed samples: {metrics['failed_samples']}")
    print(f"Character Error Rate (CER): {avg_cer:.2%}")
    print(f"Word Error Rate (WER): {avg_wer:.2%}")
    print(f"Real Time Factor (RTF): {rtf:.4f} s/sample")
    print(f"Total time: {elapsed_time:.2f} s")
    print("=" * 60)
    print(f"Results saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 60 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Fun-ASR-Nano 8kHz telephone channel model"
    )
    
    # Model configuration
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Path to model directory or ModelScope model ID')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (optional)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 precision')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 precision')
    
    # Input configuration
    parser.add_argument('--audio_file', type=str, default=None,
                        help='Single audio file for inference')
    parser.add_argument('--audio_list', type=str, default=None,
                        help='Text file with list of audio files')
    parser.add_argument('--test_data', type=str, default=None,
                        help='JSONL file for evaluation')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')
    
    # Inference configuration
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--hotwords', type=str, default=None,
                        help='Hotwords for biased decoding (comma-separated)')
    
    # JSONL keys
    parser.add_argument('--audio_key', type=str, default='source',
                        help='Key in JSONL for audio file path')
    parser.add_argument('--text_key', type=str, default='target',
                        help='Key in JSONL for reference text')
    
    # VAD configuration
    parser.add_argument('--use_vad', action='store_true',
                        help='Use VAD for long audio segmentation')
    parser.add_argument('--vad_model', type=str, default='fsmn-vad',
                        help='VAD model name')
    parser.add_argument('--max_single_segment_time', type=int, default=30000,
                        help='Max single segment time in ms (for VAD)')
    
    args = parser.parse_args()
    
    # Parse hotwords
    hotwords = None
    if args.hotwords:
        hotwords = [w.strip() for w in args.hotwords.split(',') if w.strip()]
        print(f"Hotwords: {hotwords}")
    
    # Initialize model
    print("Loading model...")
    model_kwargs = {
        'model': args.model_dir,
        'device': args.device,
        'fp16': args.fp16,
        'bf16': args.bf16,
    }
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        model_kwargs['init_param'] = args.checkpoint
        print(f"Using checkpoint: {args.checkpoint}")
    
    # Add VAD if requested
    if args.use_vad:
        model_kwargs.update({
            'vad_model': args.vad_model,
            'vad_kwargs': {'max_single_segment_time': args.max_single_segment_time},
        })
        print(f"Using VAD model: {args.vad_model}")
    
    model = AutoModel(**model_kwargs)
    print("Model loaded successfully!")
    
    # Perform inference based on input type
    if args.audio_file:
        # Single file inference
        print(f"\nInference on single file: {args.audio_file}")
        result = inference_single_file(model, args.audio_file, hotwords=hotwords)
        
        if result:
            print(f"\nRecognition result:")
            print(f"  Text: {result.get('text', '')}")
            
            # Save result
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, "result.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved to: {output_file}")
        else:
            print("Inference failed!")
            
    elif args.audio_list:
        # Batch inference
        print(f"\nBatch inference from list: {args.audio_list}")
        
        with open(args.audio_list, 'r') as f:
            audio_paths = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(audio_paths)} audio files...")
        results = inference_batch(model, audio_paths, batch_size=args.batch_size, hotwords=hotwords)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, "results.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for audio_path, result in zip(audio_paths, results):
                output = {
                    'audio': audio_path,
                    'result': result,
                }
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
        
        print(f"Results saved to: {output_file}")
        
    elif args.test_data:
        # JSONL evaluation
        print(f"\nEvaluation on JSONL: {args.test_data}")
        metrics = inference_jsonl(
            model,
            args.test_data,
            args.output_dir,
            batch_size=args.batch_size,
            hotwords=hotwords,
            audio_key=args.audio_key,
            text_key=args.text_key,
        )
        
    else:
        print("Error: Please specify --audio_file, --audio_list, or --test_data")
        sys.exit(1)


if __name__ == "__main__":
    main()
