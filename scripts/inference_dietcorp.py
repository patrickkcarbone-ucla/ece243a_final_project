#!/usr/bin/env python3
"""
DietCORP Test-Time Adaptation Inference Script.

Evaluates a trained Transformer model with DietCORP adaptation on each test trial.
Reports per-day PER to show cross-day robustness.

Usage:
    python scripts/inference_dietcorp.py \
        --checkpoint outputs/transformer/base/model_best.pt \
        --dataset dual_region_data/ptDecoder_ctc_dual_region

Or with Hydra config:
    python scripts/inference_dietcorp.py \
        --config outputs/transformer/base/.hydra/config.yaml
"""

import argparse
import os
import sys
import copy
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from edit_distance import SequenceMatcher

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural_decoder.dataset import SpeechDataset, DualRegionSpeechDataset


def load_model_and_config(checkpoint_path, config_path=None):
    """Load model from checkpoint and optionally from Hydra config."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if config_path:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = None
    
    return checkpoint, cfg


def create_model(checkpoint, cfg=None, device="cuda"):
    """Recreate model from checkpoint."""
    from neural_decoder.models.transformer import TimeMaskedTransformer
    
    # Default model params (can be overridden by config)
    model_params = {
        "neural_dim": 256,
        "n_classes": 40,
        "hidden_dim": 384,
        "num_layers": 5,
        "num_heads": 6,
        "ffn_dim": 1536,
        "patch_len": 5,
        "patch_stride": 5,
        "dropout": 0.35,
        "attn_dropout": 0.1,
        "input_dropout": 0.2,
        "max_rel_pos": 64,
        "num_masks": 20,
        "max_mask_fraction": 0.075,
        "nDays": 24,
        "device": device,
    }
    
    if cfg and "model" in cfg:
        for key, value in cfg["model"].items():
            if key in model_params:
                model_params[key] = value
    
    model = TimeMaskedTransformer(**model_params)
    
    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def apply_time_masking(x, num_masks=50, max_mask_fraction=0.375):
    """
    Apply random time-masking to input tensor.
    
    Uses zeros for masking since this operates on raw input (neural_dim),
    not the model's hidden space where the learnable mask token lives.
    """
    B, T, C = x.shape
    x_masked = x.clone()
    
    for _ in range(num_masks):
        mask_len = int(T * max_mask_fraction * torch.rand(1).item())
        if mask_len == 0:
            continue
        
        for b in range(B):
            start = torch.randint(0, max(1, T - mask_len), (1,)).item()
            end = min(start + mask_len, T)
            x_masked[b, start:end, :] = 0.0
    
    return x_masked


def dietcorp_adapt_and_infer(
    model, X, dayIdx, device, 
    num_views=64, adapt_lr=0.0001
):
    """
    Apply DietCORP adaptation to a single trial.
    
    Args:
        model: The Transformer model
        X: [1, T, C] single trial input
        dayIdx: [1] day index
        device: torch device
        num_views: Number of augmented views for adaptation
        adapt_lr: Learning rate for adaptation step
        
    Returns:
        output: Model output after adaptation
    """
    # Save original patch embedding state
    orig_state = copy.deepcopy(model.patch_embed.state_dict())
    
    # Get pseudo-labels from unadapted model
    with torch.no_grad():
        unadapted_out = model.forward(X, dayIdx)
        if isinstance(unadapted_out, dict):
            pseudo_logits = unadapted_out['phone_logits']
        else:
            pseudo_logits = unadapted_out
    
    # Create Z augmented views with heavy time-masking
    views = []
    for _ in range(num_views):
        masked_X = apply_time_masking(X, num_masks=50, max_mask_fraction=0.375)
        views.append(masked_X)
    
    views_batch = torch.cat(views, dim=0)
    dayIdx_expanded = dayIdx.expand(num_views)
    
    # Freeze all parameters except patch embedding
    for name, param in model.named_parameters():
        param.requires_grad = 'patch_embed' in name
    
    # Forward pass on augmented views
    adapted_out = model.forward(views_batch, dayIdx_expanded)
    if isinstance(adapted_out, dict):
        adapted_logits = adapted_out['phone_logits']
    else:
        adapted_logits = adapted_out
    
    # Compute pseudo-CTC loss (KL divergence)
    pseudo_probs = F.softmax(pseudo_logits.expand(num_views, -1, -1), dim=-1)
    adapt_log_probs = F.log_softmax(adapted_logits.float(), dim=-1)
    adapt_loss = F.kl_div(adapt_log_probs, pseudo_probs.detach(), reduction='batchmean')
    
    # Single gradient step
    adapt_loss.backward()
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'patch_embed' in name and param.grad is not None:
                param -= adapt_lr * param.grad
    
    # Inference with adapted model
    with torch.no_grad():
        final_out = model.forward(X, dayIdx)
    
    # Restore original state
    model.patch_embed.load_state_dict(orig_state)
    for param in model.parameters():
        param.requires_grad = True
    
    # Clear gradients
    model.zero_grad()
    
    return final_out


def evaluate_with_dietcorp(
    model, test_loader, device,
    num_views=64, adapt_lr=0.0001,
    use_dietcorp=True
):
    """
    Evaluate model on test set, optionally with DietCORP adaptation.
    
    Returns:
        overall_per: Overall phoneme error rate
        per_day_stats: Dict mapping day -> (edit_distance, seq_length, per)
    """
    model.eval()
    
    total_edit_distance = 0
    total_seq_length = 0
    per_day_stats = defaultdict(lambda: {"edit_distance": 0, "seq_length": 0})
    
    for batch_data in test_loader:
        if len(batch_data) == 6:
            X, X_44, y, X_len, y_len, dayIdx = batch_data
        else:
            X, y, X_len, y_len, dayIdx = batch_data
        
        X = X.to(device)
        y = y.to(device)
        X_len = X_len.to(device)
        y_len = y_len.to(device)
        dayIdx = dayIdx.to(device)
        
        # Compute output lengths
        patch_len = model.kernelLen
        patch_stride = model.strideLen
        output_lens = ((X_len - patch_len) / patch_stride).to(torch.int32)
        
        # Process each trial
        for i in range(X.shape[0]):
            X_single = X[i:i+1]
            dayIdx_single = dayIdx[i:i+1]
            y_single = y[i]
            y_len_single = y_len[i].item()
            day = dayIdx_single.item()
            
            if use_dietcorp:
                model_output = dietcorp_adapt_and_infer(
                    model, X_single, dayIdx_single, device,
                    num_views=num_views, adapt_lr=adapt_lr
                )
            else:
                with torch.no_grad():
                    model_output = model.forward(X_single, dayIdx_single)
            
            # Get predictions
            if isinstance(model_output, dict):
                pred = model_output['phone_logits']
            else:
                pred = model_output
            
            # Greedy decode
            seq_len = output_lens[i].item()
            if seq_len <= 0:
                continue
            
            decodedSeq = torch.argmax(pred[0, :seq_len, :], dim=-1)
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().numpy()
            decodedSeq = np.array([j for j in decodedSeq if j != 0])
            
            trueSeq = np.array(y_single[:y_len_single].cpu())
            
            matcher = SequenceMatcher(a=trueSeq.tolist(), b=decodedSeq.tolist())
            edit_dist = matcher.distance()
            
            total_edit_distance += edit_dist
            total_seq_length += len(trueSeq)
            
            per_day_stats[day]["edit_distance"] += edit_dist
            per_day_stats[day]["seq_length"] += len(trueSeq)
    
    overall_per = total_edit_distance / total_seq_length if total_seq_length > 0 else 1.0
    
    # Compute per-day PER
    for day in per_day_stats:
        stats = per_day_stats[day]
        stats["per"] = stats["edit_distance"] / stats["seq_length"] if stats["seq_length"] > 0 else 1.0
    
    return overall_per, dict(per_day_stats)


def main():
    parser = argparse.ArgumentParser(description="DietCORP Test-Time Adaptation Inference")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="dual_region_data/ptDecoder_ctc_dual_region",
        help="Path to dataset"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Optional path to Hydra config.yaml"
    )
    parser.add_argument(
        "--num_views", type=int, default=64,
        help="Number of augmented views for DietCORP"
    )
    parser.add_argument(
        "--adapt_lr", type=float, default=0.0001,
        help="Learning rate for DietCORP adaptation"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size (should be 1 for DietCORP)"
    )
    parser.add_argument(
        "--no_dietcorp", action="store_true",
        help="Disable DietCORP (baseline evaluation)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on"
    )
    args = parser.parse_args()
    
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint, cfg = load_model_and_config(args.checkpoint, args.config)
    
    print("Creating model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = create_model(checkpoint, cfg, device)
    
    print(f"Loading dataset from {args.dataset}")
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)
    
    # Use dual-region dataset if available
    if data["train"][0]["X"].shape[1] == 512:
        test_dataset = DualRegionSpeechDataset(data["test"])
    else:
        test_dataset = SpeechDataset(data["test"])
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=test_dataset.collate_fn,
    )
    
    # Baseline evaluation (no DietCORP)
    print("\n" + "=" * 60)
    print("Evaluating WITHOUT DietCORP (baseline)...")
    print("=" * 60)
    baseline_per, baseline_per_day = evaluate_with_dietcorp(
        model, test_loader, device,
        use_dietcorp=False
    )
    print(f"Baseline Overall PER: {baseline_per:.4f}")
    print("\nPer-day PER:")
    for day in sorted(baseline_per_day.keys()):
        stats = baseline_per_day[day]
        print(f"  Day {day:2d}: PER = {stats['per']:.4f} ({stats['edit_distance']}/{stats['seq_length']})")
    
    if not args.no_dietcorp:
        # DietCORP evaluation
        print("\n" + "=" * 60)
        print(f"Evaluating WITH DietCORP (views={args.num_views}, lr={args.adapt_lr})...")
        print("=" * 60)
        dietcorp_per, dietcorp_per_day = evaluate_with_dietcorp(
            model, test_loader, device,
            num_views=args.num_views,
            adapt_lr=args.adapt_lr,
            use_dietcorp=True
        )
        print(f"DietCORP Overall PER: {dietcorp_per:.4f}")
        print("\nPer-day PER:")
        for day in sorted(dietcorp_per_day.keys()):
            stats = dietcorp_per_day[day]
            baseline_day_per = baseline_per_day.get(day, {}).get("per", 0)
            improvement = baseline_day_per - stats['per']
            print(f"  Day {day:2d}: PER = {stats['per']:.4f} ({stats['edit_distance']}/{stats['seq_length']}) [Δ = {improvement:+.4f}]")
        
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Baseline PER:  {baseline_per:.4f}")
        print(f"  DietCORP PER:  {dietcorp_per:.4f}")
        print(f"  Improvement:   {baseline_per - dietcorp_per:+.4f} ({100 * (baseline_per - dietcorp_per) / baseline_per:+.1f}% relative)")
        print("=" * 60)


if __name__ == "__main__":
    main()

