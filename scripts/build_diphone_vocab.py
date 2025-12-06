#!/usr/bin/env python
"""
Build diphone vocabulary from training data.

Diphones are consecutive phoneme pairs (e.g., "SIL-AH", "AH-N", "N-SIL").
This script extracts all observed diphones from training transcripts
and builds a mapping for the diphone auxiliary loss.

Usage:
    python scripts/build_diphone_vocab.py --dataset data/ptDecoder_ctc
    python scripts/build_diphone_vocab.py --dataset dual_region_data/ptDecoder_ctc_dual_region

Output:
    data/diphone_vocab.pkl - Contains:
        - diphone_to_idx: dict mapping (phone1, phone2) -> index
        - idx_to_diphone: list of (phone1, phone2) tuples
        - n_diphones: total number of diphones (excluding blank)
        - phone_to_idx: original phoneme mapping (for reference)
        - diphone_to_phone_matrix: marginalization matrix for inference
"""

import argparse
import os
import pickle
from collections import Counter
import numpy as np


def load_dataset(dataset_path):
    """Load the pickled dataset."""
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    return data


def extract_phoneme_sequences(data):
    """Extract all phoneme sequences from training data."""
    sequences = []
    for day_data in data["train"]:
        for trial_idx in range(len(day_data["phonemes"])):
            phone_seq = day_data["phonemes"][trial_idx]
            phone_len = day_data["phoneLens"][trial_idx]
            # Get valid phonemes (up to phone_len)
            valid_seq = phone_seq[:phone_len].tolist()
            sequences.append(valid_seq)
    return sequences


def build_diphone_vocab(sequences, min_count=1):
    """
    Build diphone vocabulary from phoneme sequences.
    
    Args:
        sequences: List of phoneme sequences (each is a list of phoneme indices)
        min_count: Minimum occurrences to include a diphone
        
    Returns:
        diphone_to_idx: dict mapping (phone1, phone2) -> index
        idx_to_diphone: list of (phone1, phone2) tuples
        diphone_counts: Counter of diphone occurrences
    """
    diphone_counts = Counter()
    
    for seq in sequences:
        # Extract consecutive pairs
        for i in range(len(seq) - 1):
            diphone = (seq[i], seq[i + 1])
            diphone_counts[diphone] += 1
    
    # Filter by minimum count and sort for reproducibility
    valid_diphones = [
        dp for dp, count in diphone_counts.items() 
        if count >= min_count
    ]
    valid_diphones.sort()  # Sort for reproducibility
    
    # Build mappings (index 0 reserved for blank)
    diphone_to_idx = {dp: idx + 1 for idx, dp in enumerate(valid_diphones)}
    idx_to_diphone = [(0, 0)] + valid_diphones  # Index 0 is blank
    
    return diphone_to_idx, idx_to_diphone, diphone_counts


def build_marginalization_matrix(idx_to_diphone, n_phones):
    """
    Build matrix to marginalize diphone probabilities to phone probabilities.
    
    For each phoneme p, we sum (logsumexp) over all diphones ending with p.
    
    Args:
        idx_to_diphone: list of (phone1, phone2) tuples (index 0 is blank)
        n_phones: number of phoneme classes (excluding blank)
        
    Returns:
        matrix: [n_diphones, n_phones + 1] binary matrix
                matrix[d, p] = 1 if diphone d ends with phoneme p
    """
    n_diphones = len(idx_to_diphone)
    # +1 for blank at index 0
    matrix = np.zeros((n_diphones, n_phones + 1), dtype=np.float32)
    
    # Blank diphone maps to blank phone
    matrix[0, 0] = 1.0
    
    # Each diphone maps to its ending phoneme
    for d_idx, (p1, p2) in enumerate(idx_to_diphone):
        if d_idx == 0:
            continue  # Skip blank
        # p2 is the ending phoneme (1-indexed in phoneme space, but diphones use 0-indexed)
        # The phone indices in the data are 1-indexed (1-40), so we map directly
        if p2 >= 0 and p2 <= n_phones:
            matrix[d_idx, p2] = 1.0
    
    return matrix


def main():
    parser = argparse.ArgumentParser(description="Build diphone vocabulary")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="dual_region_data/ptDecoder_ctc_dual_region",
        help="Path to dataset pickle file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/diphone_vocab.pkl",
        help="Output path for diphone vocabulary"
    )
    parser.add_argument(
        "--min-count", 
        type=int, 
        default=1,
        help="Minimum occurrences to include a diphone"
    )
    parser.add_argument(
        "--n-phones",
        type=int,
        default=40,
        help="Number of phoneme classes (excluding blank)"
    )
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.dataset}...")
    data = load_dataset(args.dataset)
    
    print("Extracting phoneme sequences from training data...")
    sequences = extract_phoneme_sequences(data)
    print(f"  Found {len(sequences)} training trials")
    
    # Collect stats on phoneme usage
    all_phones = set()
    for seq in sequences:
        all_phones.update(seq)
    print(f"  Unique phonemes observed: {len(all_phones)}")
    print(f"  Phoneme range: {min(all_phones)} to {max(all_phones)}")
    
    print(f"\nBuilding diphone vocabulary (min_count={args.min_count})...")
    diphone_to_idx, idx_to_diphone, diphone_counts = build_diphone_vocab(
        sequences, min_count=args.min_count
    )
    
    n_diphones = len(idx_to_diphone) - 1  # Exclude blank
    print(f"  Total unique diphones: {n_diphones}")
    print(f"  Total diphone occurrences: {sum(diphone_counts.values())}")
    
    # Show most common diphones
    print("\n  Top 10 most common diphones:")
    for dp, count in diphone_counts.most_common(10):
        print(f"    {dp}: {count}")
    
    print("\nBuilding marginalization matrix...")
    margin_matrix = build_marginalization_matrix(idx_to_diphone, args.n_phones)
    print(f"  Matrix shape: {margin_matrix.shape}")
    
    # Verify each diphone maps to exactly one phone
    row_sums = margin_matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Each diphone should map to exactly one phone"
    
    # Save vocabulary
    vocab = {
        "diphone_to_idx": diphone_to_idx,
        "idx_to_diphone": idx_to_diphone,
        "n_diphones": n_diphones,
        "n_phones": args.n_phones,
        "diphone_counts": dict(diphone_counts),
        "diphone_to_phone_matrix": margin_matrix,
    }
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(vocab, f)
    
    print(f"\nSaved diphone vocabulary to {args.output}")
    print(f"  n_diphones (excl. blank): {n_diphones}")
    print(f"  Total diphone classes (incl. blank): {n_diphones + 1}")


if __name__ == "__main__":
    main()

