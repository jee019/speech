#!/usr/bin/env python3
"""
ENG 데이터용 샘플 선정 스크립트
Select 15 samples from 200 ENG samples for MOS/SMOS evaluation.

선정 기준:
1. 화자 다양성: 가능한 한 많은 서로 다른 화자 포함 (LibriSpeech 형식)
2. 길이 다양성: 짧음/중간/김 균등 분포 (각 5개씩)

참고: ENG 데이터는 LibriSpeech 형식 (260-123286-0009.wav)이므로
      파일명에서 화자 ID를 추출하여 화자 다양성을 확보합니다.
모든 모델에서 동일한 15개 파일명이 사용됩니다.
"""

import json
import random
import numpy as np
import csv
import os
from pathlib import Path
from collections import Counter

def extract_speaker_id(filename):
    """
    Extract speaker ID from LibriSpeech format filename.
    
    LibriSpeech format: "260-123286-0009.wav" -> "260"
    """
    name = filename.replace('.wav', '').replace('.WAV', '')
    
    # LibriSpeech format: SPEAKER_ID-BOOK_ID-CHAPTER_ID
    if '-' in name:
        return name.split('-')[0]
    
    # Fallback: use first part of filename
    return name.split('_')[0] if '_' in name else name[:6]

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
    Select n samples using stratified sampling based on speaker and length.
    
    선정 기준:
    1. 화자 다양성: 가능한 한 많은 서로 다른 화자 포함
    2. 길이 다양성: 짧음/중간/김 균등 분포
    
    핵심 로직:
    1. 화자별로 그룹화
    2. 길이별로 그룹화 (short/mid/long, 33/34/33 percentile)
    3. 각 화자에서 최대 1-2개만 선택하여 다양성 확보
    4. 각 길이 카테고리에서 5개씩 균등 선택
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Step 1: Add metadata to each sample
    for i, s in enumerate(samples):
        s['_index'] = i
        s['_transcript_len'] = len(s['transcript'])
        s['_speaker_id'] = extract_speaker_id(s['audio_fname'])
    
    # Step 2: Calculate length percentiles for stratification
    len_scores = [s['_transcript_len'] for s in samples]
    len_33 = np.percentile(len_scores, 33)
    len_67 = np.percentile(len_scores, 67)
    
    # Step 3: Categorize samples by length and group by speaker
    length_groups = {'short': [], 'mid': [], 'long': []}
    speaker_groups = {}
    
    for s in samples:
        # Length category
        if s['_transcript_len'] < len_33:
            s['_len_category'] = 'short'
            length_groups['short'].append(s)
        elif s['_transcript_len'] < len_67:
            s['_len_category'] = 'mid'
            length_groups['mid'].append(s)
        else:
            s['_len_category'] = 'long'
            length_groups['long'].append(s)
        
        # Speaker group
        speaker_id = s['_speaker_id']
        if speaker_id not in speaker_groups:
            speaker_groups[speaker_id] = []
        speaker_groups[speaker_id].append(s)
    
    # Step 4: Select samples ensuring both speaker and length diversity
    selected = []
    selected_speakers = Counter()
    selected_lengths = Counter()
    
    # Target: 각 길이 카테고리에서 5개씩 (총 15개)
    target_per_length = {'short': 5, 'mid': 5, 'long': 5}
    
    # 화자당 최대 샘플 수: 1개를 우선, 필요시 2개까지
    max_samples_per_speaker = 2
    
    # Shuffle speakers for randomness
    all_speakers = list(speaker_groups.keys())
    random.shuffle(all_speakers)
    
    # First pass: Select samples ensuring both speaker and length diversity
    length_needs = target_per_length.copy()
    
    # Sort samples by length within each speaker group
    for speaker_id in all_speakers:
        speaker_groups[speaker_id].sort(key=lambda x: x['_transcript_len'])
    
    # Select samples ensuring both speaker and length diversity
    for speaker_id in all_speakers:
        if len(selected) >= n:
            break
        
        # Check if we can add more samples from this speaker
        if selected_speakers[speaker_id] >= max_samples_per_speaker:
            continue
        
        # Find which length categories still need samples
        needed_categories = [cat for cat, count in length_needs.items() if count > 0]
        if not needed_categories:
            # If all length categories are filled, pick any
            needed_categories = ['short', 'mid', 'long']
        
        # Try to find a sample in a needed length category
        for length_cat in needed_categories:
            if len(selected) >= n:
                break
            
            available = [s for s in speaker_groups[speaker_id] 
                        if s['_len_category'] == length_cat and s not in selected]
            
            if available:
                # Select middle sample for length diversity
                selected_sample = available[len(available)//2]
                selected.append(selected_sample)
                selected_speakers[speaker_id] += 1
                selected_lengths[length_cat] += 1
                length_needs[length_cat] -= 1
                break
    
    # Second pass: Fill remaining slots
    # Prioritize underrepresented length categories and speakers
    while len(selected) < n:
        all_available = [s for s in samples if s not in selected]
        if not all_available:
            break
        
        # Find which length categories need more samples
        length_priorities = {}
        for cat in ['short', 'mid', 'long']:
            current_count = selected_lengths[cat]
            target_count = target_per_length[cat]
            length_priorities[cat] = target_count - current_count
        
        # Sort available samples by priority:
        # 1. Length category need (higher priority for underrepresented)
        # 2. Speaker diversity (lower count = higher priority)
        def priority_score(s):
            len_priority = length_priorities.get(s['_len_category'], 0)
            speaker_count = selected_speakers[s['_speaker_id']]
            return (len_priority, -speaker_count)  # Higher is better
        
        all_available.sort(key=priority_score, reverse=True)
        
        candidate = all_available[0]
        selected.append(candidate)
        selected_speakers[candidate['_speaker_id']] += 1
        selected_lengths[candidate['_len_category']] += 1
    
    # Step 5: Shuffle final selection
    random.shuffle(selected)
    
    return selected[:n], selected_speakers

def create_nmos_csv(selected_indices, models, output_file, base_dir, selected_transcripts):
    """
    Create NMOS.csv with randomly shuffled samples from all models.
    Each selected transcript appears once per model, but order is randomized.
    동일한 transcript를 읽는 음성은 연속 배치되지 않도록 처리.
    
    결과:
    - 총 15개 transcript × 모델 수 = 샘플 수
    - 순서는 완전히 랜덤 (모델 정보 숨김)
    - 동일 transcript 연속 배치 방지
    - CSV에 transcript 정보 포함 (HTML에는 표시 안 함)
    """
    # Collect all samples with transcript info
    all_samples = []
    for model in models:
        meta_file = Path(base_dir) / model / "meta_data.jsonl"
        with open(meta_file, 'r', encoding='utf-8') as f:
            samples = [json.loads(l) for l in f]
        
        for idx in selected_indices:
            if idx < len(samples):
                audio_fname = samples[idx]['audio_fname']
                transcript = samples[idx].get('transcript', '')
                audio_path = f"wav/eval_data/eng/{model}/{audio_fname}"
                all_samples.append((model, idx, audio_path, transcript))
    
    # Shuffle with constraint: 같은 transcript가 연속되지 않도록
    shuffled = []
    remaining = all_samples.copy()
    random.shuffle(remaining)
    
    last_transcript = None
    max_attempts = len(all_samples) * 10  # 최대 시도 횟수
    
    while remaining and len(shuffled) < len(all_samples) and max_attempts > 0:
        max_attempts -= 1
        found = False
        
        # 같은 transcript가 아닌 샘플 찾기
        for i, sample in enumerate(remaining):
            _, _, _, transcript = sample
            if transcript != last_transcript:
                shuffled.append(sample)
                remaining.pop(i)
                last_transcript = transcript
                found = True
                break
        
        # 같은 transcript가 아닌 샘플을 찾지 못한 경우, 그냥 추가
        if not found and remaining:
            sample = remaining.pop(0)
            shuffled.append(sample)
            last_transcript = sample[3]
    
    # 남은 샘플 추가
    shuffled.extend(remaining)
    
    # Write to CSV with transcript column
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order', 'filename', 'transcript'])
        for order, (model, idx, audio_path, transcript) in enumerate(shuffled, start=1):
            writer.writerow([order, audio_path, transcript])
    
    # 연속 배치 확인
    consecutive_count = 0
    for i in range(len(shuffled) - 1):
        if shuffled[i][3] == shuffled[i+1][3]:
            consecutive_count += 1
    
    print(f"Created {output_file} with {len(shuffled)} samples")
    if consecutive_count > 0:
        print(f"  ⚠️ Warning: {consecutive_count} consecutive same transcripts detected")
    else:
        print(f"  ✅ No consecutive same transcripts")

def create_smos_csv(selected_indices, models, output_file, base_dir, selected_transcripts):
    """
    Create SMOS.csv with pairs of (generated_audio, reference_speaker_audio).
    For each selected transcript, create pairs (one per model).
    
    결과:
    - 총 15개 transcript × 모델 수 = 쌍 수
    - 각 쌍: (생성된 오디오, 참조 화자 오디오)
    - 순서는 랜덤
    - CSV에 transcript 정보 포함
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
        # Get transcript from first model (all models should have same transcript at same index)
        transcript = model_metadata[models[0]][idx].get('transcript', '') if idx < len(model_metadata[models[0]]) else ''
        
        for model in models:
            if idx < len(model_metadata[model]):
                sample = model_metadata[model][idx]
                audio_fname = sample['audio_fname']
                ref_speaker_fname = sample['ref_speaker_fname']
                
                generated_path = f"wav/eval_data/eng/{model}/{audio_fname}"
                reference_path = f"wav/eval_data/eng/speakers/{ref_speaker_fname}"
                
                rows.append((generated_path, reference_path, transcript))
    
    # Shuffle rows to randomize order
    random.shuffle(rows)
    
    # Write to CSV with transcript column
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['order', 'filename1', 'filename2', 'transcript'])
        for order, (gen_path, ref_path, transcript) in enumerate(rows, start=1):
            writer.writerow([order, gen_path, ref_path, transcript])
    
    print(f"Created {output_file} with {len(rows)} samples")

def main():
    # Configuration for ENG
    base_dir = Path("wav/eval_data/eng")
    models = ['mimi', 'lmspt-dc', 'lmspt-mimi', 'dualcodec', 'cosy2']  # lmspt-mimi 추가
    n_samples = 15
    seed = 42
    
    print("=" * 70)
    print("ENG Sample Selection for MOS/SMOS Evaluation")
    print("=" * 70)
    print(f"Selecting {n_samples} samples from 200 available ENG samples")
    print(f"Models: {', '.join(models)}")
    print()
    print("선정 기준:")
    print("  1. 화자 다양성: 가능한 한 많은 서로 다른 화자 포함 (LibriSpeech 형식)")
    print("  2. 길이 다양성: 짧음/중간/김 균등 분포 (각 5개씩)")
    print()
    print("참고: ENG 데이터는 LibriSpeech 형식 (260-123286-0009.wav)이므로")
    print("      파일명에서 화자 ID를 추출하여 화자 다양성을 확보합니다.")
    print()
    print("중요: 모든 모델에서 동일한 15개 파일명이 사용됩니다.")
    print("      (각 모델의 동일 인덱스에 같은 파일명이 위치)")
    print()
    
    # Load metadata from first model (all models have same structure)
    model_dir = base_dir / models[0]
    all_samples = load_metadata(model_dir)
    print(f"Loaded {len(all_samples)} samples from {models[0]}")
    
    # Extract speaker information
    speaker_ids = [extract_speaker_id(s['audio_fname']) for s in all_samples]
    unique_speakers = len(set(speaker_ids))
    print(f"Total unique speakers: {unique_speakers}")
    print()
    
    # Select samples using stratified sampling
    selected_samples, speaker_distribution = stratified_sample(all_samples, n=n_samples, seed=seed)
    selected_indices = [s['_index'] for s in selected_samples]
    selected_transcripts = [s['transcript'] for s in selected_samples]
    
    print(f"Selected {len(selected_samples)} samples:")
    print("-" * 70)
    for i, s in enumerate(selected_samples, 1):
        speaker_id = s['_speaker_id']
        print(f"{i:2d}. {s['audio_fname']:35s} | "
              f"Speaker: {speaker_id:8s} | "
              f"Len: {s['_transcript_len']:3d} ({s['_len_category']:5s}) | "
              f"Index: {s['_index']:3d}")
    
    # Statistics
    print("\n" + "=" * 70)
    print("Selection Statistics:")
    print("-" * 70)
    len_vals = [s['_transcript_len'] for s in selected_samples]
    selected_speaker_ids = [s['_speaker_id'] for s in selected_samples]
    unique_selected_speakers = len(set(selected_speaker_ids))
    
    # Length distribution
    from collections import Counter
    len_categories = Counter([s['_len_category'] for s in selected_samples])
    
    print(f"Length:        {min(len_vals):2d} - {max(len_vals):2d} (avg: {np.mean(len_vals):.1f})")
    print(f"Length distribution: Short={len_categories.get('short', 0)}, Mid={len_categories.get('mid', 0)}, Long={len_categories.get('long', 0)}")
    print(f"Unique Speakers: {unique_selected_speakers} out of {unique_speakers} ({unique_selected_speakers/unique_speakers*100:.1f}%)")
    
    # Speaker distribution
    print("\nSpeaker Distribution:")
    for speaker_id, count in sorted(speaker_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {speaker_id}: {count} sample(s)")
    
    # Create CSV files
    print("\n" + "=" * 70)
    print("Creating CSV files...")
    print("-" * 70)
    
    output_dir = Path("filelist")
    output_dir.mkdir(exist_ok=True)
    
    create_nmos_csv(selected_indices, models, output_dir / "NMOS_eng.csv", base_dir, selected_transcripts)
    create_smos_csv(selected_indices, models, output_dir / "SMOS_eng.csv", base_dir, selected_transcripts)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review the selected samples above")
    print(f"2. Run: python render_eval.py (or modify to use eng CSV files)")
    print(f"3. Open eval.html in a browser")

if __name__ == "__main__":
    main()
