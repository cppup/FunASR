#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Keyword-focused evaluation for telephone outbound ASR.
Calculates Keyword Error Rate (KWER) and overall CER/WER.
"""

import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def compute_cer(ref, hyp):
    """Compute Character Error Rate."""
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    # Levenshtein distance
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
    """Compute Word Error Rate."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
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


def compute_kwer(ref, hyp, keywords):
    """
    Compute Keyword Error Rate (KWER).
    
    Args:
        ref: Reference text
        hyp: Hypothesis text
        keywords: List of keywords to track
    
    Returns:
        Tuple of (kwer, keyword_stats)
    """
    keyword_stats = {kw: {'total': 0, 'correct': 0, 'error': 0} for kw in keywords}
    
    total_kw = 0
    correct_kw = 0
    
    for kw in keywords:
        # Count occurrences in reference
        ref_count = ref.count(kw)
        hyp_count = hyp.count(kw)
        
        keyword_stats[kw]['total'] += ref_count
        keyword_stats[kw]['correct'] += min(ref_count, hyp_count)
        keyword_stats[kw]['error'] += abs(ref_count - hyp_count) + max(0, ref_count - hyp_count)
        
        total_kw += ref_count
        correct_kw += min(ref_count, hyp_count)
    
    kwer = 1.0 - (correct_kw / total_kw) if total_kw > 0 else 0.0
    
    return kwer, keyword_stats


def load_keywords(keyword_file):
    """Load keywords from file (one keyword per line)."""
    keywords = []
    with open(keyword_file, 'r', encoding='utf-8') as f:
        for line in f:
            kw = line.strip()
            if kw:
                keywords.append(kw)
    return keywords


def evaluate_results(results_jsonl, keyword_file=None):
    """
    Evaluate ASR results with focus on keywords.
    
    Args:
        results_jsonl: Path to results JSONL file (from inference_8k.py)
        keyword_file: Path to keyword file (optional)
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load keywords
    keywords = []
    if keyword_file:
        keywords = load_keywords(keyword_file)
        print(f"Loaded {len(keywords)} keywords from {keyword_file}")
    
    # Load results
    results = []
    with open(results_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"Loaded {len(results)} results from {results_jsonl}")
    
    # Calculate metrics
    total_cer = 0.0
    total_wer = 0.0
    total_kwer = 0.0
    valid_count = 0
    
    keyword_stats_global = defaultdict(lambda: {'total': 0, 'correct': 0, 'error': 0})
    
    for result in results:
        ref = result.get('reference', '')
        hyp = result.get('hypothesis', '')
        
        if not ref:
            continue
        
        # CER/WER
        cer = compute_cer(ref, hyp)
        wer = compute_wer(ref, hyp)
        
        total_cer += cer
        total_wer += wer
        valid_count += 1
        
        # KWER
        if keywords:
            kwer, kw_stats = compute_kwer(ref, hyp, keywords)
            total_kwer += kwer
            
            for kw, stats in kw_stats.items():
                keyword_stats_global[kw]['total'] += stats['total']
                keyword_stats_global[kw]['correct'] += stats['correct']
                keyword_stats_global[kw]['error'] += stats['error']
    
    # Calculate averages
    avg_cer = total_cer / valid_count if valid_count > 0 else 0.0
    avg_wer = total_wer / valid_count if valid_count > 0 else 0.0
    avg_kwer = total_kwer / valid_count if valid_count > 0 else 0.0
    
    # Calculate per-keyword accuracy
    keyword_accuracy = {}
    for kw, stats in keyword_stats_global.items():
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total']
            keyword_accuracy[kw] = {
                'accuracy': acc,
                'total': stats['total'],
                'correct': stats['correct'],
                'error': stats['error']
            }
    
    metrics = {
        'total_samples': len(results),
        'valid_samples': valid_count,
        'cer': avg_cer,
        'wer': avg_wer,
        'kwer': avg_kwer,
        'keyword_accuracy': keyword_accuracy
    }
    
    return metrics


def print_report(metrics):
    """Print evaluation report."""
    print("\n" + "="*80)
    print("ASR Evaluation Report")
    print("="*80)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Valid samples: {metrics['valid_samples']}")
    print(f"\nOverall Metrics:")
    print(f"  Character Error Rate (CER): {metrics['cer']:.2%}")
    print(f"  Word Error Rate (WER): {metrics['wer']:.2%}")
    
    if metrics['kwer'] > 0:
        print(f"  Keyword Error Rate (KWER): {metrics['kwer']:.2%}")
        
        print(f"\nPer-Keyword Accuracy:")
        print(f"  {'Keyword':<20} {'Accuracy':>10} {'Total':>8} {'Correct':>8} {'Error':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        
        # Sort by accuracy (ascending) to highlight problematic keywords
        sorted_kw = sorted(
            metrics['keyword_accuracy'].items(),
            key=lambda x: x[1]['accuracy']
        )
        
        for kw, stats in sorted_kw:
            print(f"  {kw:<20} {stats['accuracy']:>10.2%} {stats['total']:>8} "
                  f"{stats['correct']:>8} {stats['error']:>8}")
        
        # Calculate macro-average keyword accuracy
        if len(sorted_kw) > 0:
            macro_avg = np.mean([stats['accuracy'] for _, stats in sorted_kw])
            print(f"\n  {'Macro-Average':<20} {macro_avg:>10.2%}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Keyword-focused evaluation for telephone ASR"
    )
    
    parser.add_argument('--results', type=str, required=True,
                        help='Path to results JSONL file (from inference_8k.py)')
    parser.add_argument('--keywords', type=str, default=None,
                        help='Path to keyword file (one keyword per line)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output metrics JSON file')
    
    args = parser.parse_args()
    
    # Evaluate
    metrics = evaluate_results(args.results, args.keywords)
    
    # Print report
    print_report(metrics)
    
    # Save metrics
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
