#!/usr/bin/env python3
"""
KOR 데이터용 샘플 선정 스크립트
Select 15 samples from 200 KOR samples for MOS/SMOS evaluation.

선정 기준:
1. 길이 다양성: 짧음/중간/김 균등 분포 (각 5개씩)

참고: KOR 데이터는 파일명에 화자 정보가 명확하지 않으므로 화자 다양성은 고려하지 않습니다.
모든 모델에서 동일한 15개 파일명이 사용됩니다.
"""

import json
import random
import numpy as np
import csv
import os
from pathlib import Path
from collections import Counter

def load_metadata(model_dir):
    """Load metadata from a model directory."""
    meta_file = Path(model_dir) / "meta_data.jsonl"
    samples = []
    with open(meta_file, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples

def stratified_sample(samples, n=15, seed=42):
    """
    Select n samples using stratified sampling based on transcript length only.
    
    선정 기준:
    - 길이 다양성: 짧음/중간/김 균등 분포
    
    핵심 로직:
    1. 길이별로 그룹화 (short/mid/long, 33/34/33 percentile)
    2. 각 길이 카테고리에서 5개씩 균등 선택
    3. 무작위로 섞기
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Step 1: Add metadata to each sample
    for i, s in enumerate(samples):
        s['_index'] = i
        s['_transcript_len'] = len(s['transcript'])
    
    # Step 2: Calculate length percentiles for stratification
    len_scores = [s['_transcript_len'] for s in samples]
    len_33 = np.percentile(len_scores, 33)
    len_67 = np.percentile(len_scores, 67)
    
    # Step 3: Categorize samples by length
    length_groups = {'short': [], 'mid': [], 'long': []}
    for s in samples:
        if s['_transcript_len'] < len_33:
            s['_len_category'] = 'short'
            length_groups['short'].append(s)
        elif s['_transcript_len'] < len_67:
            s['_len_category'] = 'mid'
            length_groups['mid'].append(s)
        else:
            s['_len_category'] = 'long'
            length_groups['long'].append(s)
    
    # Step 4: Select 5 samples from each length category
    selected = []
    samples_per_category = n // 3  # 5 samples per category
    
    for category in ['short', 'mid', 'long']:
        category_samples = length_groups[category]
        if len(category_samples) == 0:
            continue
        
        # Sort by length for even distribution
        category_samples.sort(key=lambda x: x['_transcript_len'])
        
        # Select evenly spaced samples
        if len(category_samples) <= samples_per_category:
            selected.extend(category_samples)
        else:
            indices = np.linspace(0, len(category_samples)-1, samples_per_category, dtype=int)
            for idx in indices:
                selected.append(category_samples[idx])
    
    # Step 5: Shuffle final selection
    random.shuffle(selected)
    
    return selected[:n]

def create_nmos_csv(selected_indices, models, output_file, base_dir):
    """
    Create NMOS.csv with randomly shuffled samples from all models.
    Each selected transcript appears once per model, but order is randomized.
    
    결과:
    - 총 15개 transcript × 4개 모델 = 60개 샘플
    - 순서는 완전히 랜덤 (모델 정보 숨김)
    """
    rows = []
    
    # Collect all samples (model, index, audio_fname)
    all_samples = []
    for model in models:
        meta_file = Path(base_dir) / model / "meta_data.jsonl"
        with open(meta_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(l) for l in f]
        
        for idx in selected_indices:
            if idx < len(samples):
                audio_fname = samples[idx]['audio_fname']
                audio_path = f"wav/eval_data/kor/{model}/{audio_fname}"
                all_samples.append((model, idx, audio_path))
    
    # Shuffle all samples to randomize order
    random.shuffle(all_samples)
    
    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order', 'filename'])
        for order, (model, idx, audio_path) in enumerate(all_samples, start=1):
            writer.writerow([order, audio_path])
    
    print(f"Created {output_file} with {len(all_samples)} samples")

def create_smos_csv(selected_indices, models, output_file, base_dir):
    """
    Create SMOS.csv with pairs of (generated_audio, reference_speaker_audio).
    For each selected transcript, create 4 pairs (one per model).
    
    결과:
    - 총 15개 transcript × 4개 모델 = 60개 쌍
    - 각 쌍: (생성된 오디오, 참조 화자 오디오)
    - 순서는 랜덤
    """
    rows = []
    
    # Load metadata for all models
    model_metadata = {}
    for model in models:
        meta_file = Path(base_dir) / model / "meta_data.jsonl"
        with open(meta_file, 'r', encoding='utf-8') as f:
            model_metadata[model] = [json.loads(l) for l in f]
    
    # For each selected index, create pairs for all models
    for idx in selected_indices:
        for model in models:
            if idx < len(model_metadata[model]):
                sample = model_metadata[model][idx]
                audio_fname = sample['audio_fname']
                ref_speaker_fname = sample['ref_speaker_fname']
                
                generated_path = f"wav/eval_data/kor/{model}/{audio_fname}"
                reference_path = f"wav/eval_data/kor/speakers/{ref_speaker_fname}"
                
                rows.append((generated_path, reference_path))
    
    # Shuffle rows to randomize order
    random.shuffle(rows)
    
    # Write to CSV
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order', 'filename1', 'filename2'])
        for order, (gen_path, ref_path) in enumerate(rows, start=1):
            writer.writerow([order, gen_path, ref_path])
    
    print(f"Created {output_file} with {len(rows)} samples")

def main():
    # Configuration for KOR
    base_dir = Path("wav/eval_data/kor")
    models = ['mimi', 'lmspt-dc', 'dualcodec', 'cosy2']
    n_samples = 15
    seed = 42
    
    print("=" * 70)
    print("KOR Sample Selection for MOS/SMOS Evaluation")
    print("=" * 70)
    print(f"Selecting {n_samples} samples from 200 available KOR samples")
    print(f"Models: {', '.join(models)}")
    print()
    print("선정 기준:")
    print("  1. 길이 다양성: 짧음/중간/김 균등 분포 (각 5개씩)")
    print()
    print("참고: KOR 데이터는 파일명에 화자 정보가 명확하지 않으므로")
    print("      화자 다양성은 고려하지 않습니다.")
    print()
    print("중요: 모든 모델에서 동일한 15개 파일명이 사용됩니다.")
    print("      (각 모델의 동일 인덱스에 같은 파일명이 위치)")
    print()
    
    # Load metadata from first model (all models have same structure)
    model_dir = base_dir / models[0]
    all_samples = load_metadata(model_dir)
    print(f"Loaded {len(all_samples)} samples from {models[0]}")
    print()
    
    # Select samples using stratified sampling
    selected_samples = stratified_sample(all_samples, n=n_samples, seed=seed)
    selected_indices = [s['_index'] for s in selected_samples]
    
    print(f"Selected {len(selected_samples)} samples:")
    print("-" * 70)
    for i, s in enumerate(selected_samples, 1):
        print(f"{i:2d}. {s['audio_fname']:35s} | "
              f"Len: {s['_transcript_len']:3d} ({s['_len_category']:5s}) | "
              f"Index: {s['_index']:3d}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Selection Statistics:")
    print("-" * 70)
    len_vals = [s['_transcript_len'] for s in selected_samples]
    
    # Length distribution
    from collections import Counter
    len_categories = Counter([s['_len_category'] for s in selected_samples])
    
    print(f"Length:        {min(len_vals):2d} - {max(len_vals):2d} (avg: {np.mean(len_vals):.1f})")
    print(f"Length distribution: Short={len_categories.get('short', 0)}, Mid={len_categories.get('mid', 0)}, Long={len_categories.get('long', 0)}")
    
    # Create CSV files
    print("\n" + "=" * 70)
    print("Creating CSV files...")
    print("-" * 70)
    
    output_dir = Path("filelist")
    output_dir.mkdir(exist_ok=True)
    
    create_nmos_csv(selected_indices, models, output_dir / "NMOS_kor.csv", base_dir)
    create_smos_csv(selected_indices, models, output_dir / "SMOS_kor.csv", base_dir)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the selected samples above")
    print(f"2. Run: python render_eval.py (or modify to use kor CSV files)")
    print(f"3. Open eval.html in a browser")

if __name__ == "__main__":
    main()
