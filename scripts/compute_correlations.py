"""
Compute correlations between Area 6v and Area 44 channels.

This script precomputes the Pearson correlation between each 6v channel and each 44 channel
across all training data. The output is used by CorrelationGuidedDropout to identify
which 6v channels are likely contaminated by shared noise.

Usage:
    python scripts/compute_correlations.py

Output:
    data/6v_44_correlations.npy - [256, 256] correlation matrix
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def compute_correlations(dataset_path, output_path):
    """
    Compute correlations between 6v and 44 channels.
    
    Args:
        dataset_path: Path to dual-region dataset pickle
        output_path: Path to save correlation matrix
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, "rb") as f:
        data = pickle.load(f)
    
    # Concatenate all training data
    all_6v = []
    all_44 = []
    
    for day_data in data["train"]:
        for trial in day_data["sentenceDat"]:
            # trial shape: [T, 512]
            if trial.shape[1] != 512:
                raise ValueError(f"Expected 512 channels, got {trial.shape[1]}")
            
            all_6v.append(trial[:, :256])   # Area 6v
            all_44.append(trial[:, 256:])   # Area 44
    
    # Stack all data: [total_time, 256]
    all_6v = np.concatenate(all_6v, axis=0)
    all_44 = np.concatenate(all_44, axis=0)
    
    print(f"Computing correlations from {all_6v.shape[0]} time points...")
    
    # Compute correlation matrix [256 x 256]
    # corr[i, j] = correlation between 6v channel i and 44 channel j
    n_channels = 256
    corr_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(n_channels):
            corr = np.corrcoef(all_6v[:, i], all_44[:, j])[0, 1]
            corr_matrix[i, j] = corr
        
        if (i + 1) % 32 == 0:
            print(f"  Processed {i + 1}/{n_channels} 6v channels...")
    
    # Save correlation matrix
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, corr_matrix)
    print(f"Saved correlation matrix to {output_path}")
    
    # Print summary statistics
    max_corr_per_6v = np.abs(corr_matrix).max(axis=1)
    print(f"\nSummary:")
    print(f"  Mean max |correlation| per 6v channel: {max_corr_per_6v.mean():.4f}")
    print(f"  Max |correlation| overall: {np.abs(corr_matrix).max():.4f}")
    print(f"  Channels with max |corr| > 0.3: {(max_corr_per_6v > 0.3).sum()}")
    print(f"  Channels with max |corr| > 0.2: {(max_corr_per_6v > 0.2).sum()}")
    print(f"  Channels with max |corr| > 0.1: {(max_corr_per_6v > 0.1).sum()}")
    
    return corr_matrix


if __name__ == "__main__":
    # Default paths
    dataset_path = BASE_DIR / "dual_region_data" / "ptDecoder_ctc_dual_region"
    output_path = BASE_DIR / "data" / "6v_44_correlations.npy"
    
    # Check if dataset exists at primary location
    if not dataset_path.exists():
        # Try alternative locations
        alt_paths = [
            BASE_DIR / "data" / "DualRegionData" / "ptDecoder_ctc_dual_region",
            BASE_DIR / "data" / "ptDecoder_ctc_dual_region",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                dataset_path = alt_path
                break
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run formatCompetitionDualData.ipynb first to create the dual-region dataset.")
        sys.exit(1)
    
    compute_correlations(str(dataset_path), str(output_path))

