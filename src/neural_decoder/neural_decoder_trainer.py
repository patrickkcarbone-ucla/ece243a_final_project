import os
import pickle
import time
import json
import csv
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from neural_decoder.dataset import SpeechDataset, DualRegionSpeechDataset


def _padding_single_region(batch):
    """Padding function for single-region (5-item) batches."""
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def _padding_dual_region(batch):
    """Padding function for dual-region (6-item) batches."""
    X_6v, X_44, y, X_lens, y_lens, days = zip(*batch)
    X_6v_padded = pad_sequence(X_6v, batch_first=True, padding_value=0)
    X_44_padded = pad_sequence(X_44, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_6v_padded,
        X_44_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def getDatasetLoaders(
    datasetName,
    batchSize,
    use_dual_region=False,
):
    """
    Create train and test dataloaders.
    
    Args:
        datasetName: Path to pickled dataset
        batchSize: Batch size
        use_dual_region: If True, use DualRegionSpeechDataset (returns 6v and 44 separately)
    """
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    # Detect if data has 512 channels (dual-region)
    sample_channels = loadedData["train"][0]["sentenceDat"][0].shape[1]
    is_512_channel_data = sample_channels == 512
    
    if use_dual_region:
        if not is_512_channel_data:
            raise ValueError(
                f"use_dual_region=True but data has {sample_channels} channels. "
                "Dual-region mode requires 512-channel data."
            )
        DatasetClass = DualRegionSpeechDataset
        collate_fn = _padding_dual_region
        print(f"Using DualRegionSpeechDataset (512 channels -> 6v + 44 split)")
    else:
        DatasetClass = SpeechDataset
        collate_fn = _padding_single_region
        print(f"Using SpeechDataset ({sample_channels} channels)")

    train_ds = DatasetClass(loadedData["train"], transform=None)
    test_ds = DatasetClass(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, test_loader, loadedData


def trainModel(cfg: DictConfig):
    os.makedirs(cfg.outputDir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save training arguments (hyperparameters) in the original binary pickle format
    with open(os.path.join(cfg.outputDir, "args"), "wb") as file:
        pickle.dump(cfg, file)

    # Additionally save arguments in a human-readable JSON format for easier inspection
    try:
        args_for_json = OmegaConf.to_container(cfg, resolve=True)

        with open(os.path.join(cfg.outputDir, "args.json"), "w") as f:
            json.dump(args_for_json, f, indent=2)
    except Exception as e:
        # Don't let logging failures break training
        print(f"Warning: could not save args.json: {e}")

    # Check if using dual-region augmentation (requires splitting 6v and 44)
    use_dual_region = getattr(cfg, "useDualRegion", False)
    
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        cfg.datasetPath,
        cfg.batchSize,
        use_dual_region=use_dual_region,
    )

    # Instantiate model, optimizer, and augmentations directly from Hydra configs
    # Exclude 'name' from being passed to the model constructor (it's only for directory naming)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)  # Remove 'name' if present
    model = hydra.utils.instantiate(
        model_cfg,
        nDays=len(loadedData["train"]),
        device=device,
    ).to(device)

    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.parameters(),
    )

    augmenter = hydra.utils.instantiate(cfg.augmentations).to(device)

    # Create scheduler based on training mode
    # Epoch-based (Transformer): LR decay handled in trainer, use dummy scheduler
    # Batch-based (GRU): Use LinearLR scheduler
    if hasattr(cfg, "nEpochs") and cfg.nEpochs is not None:
        # Epoch-based mode: scheduler not used (trainer handles LR decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: 1.0
        )
    else:
        # Batch-based mode: linear LR decay from lrStart to lrEnd
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=cfg.lrEnd / cfg.lrStart,
            total_iters=cfg.nBatch,
        )

    trainer = hydra.utils.instantiate(
        cfg.trainer,
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        augmenter=augmenter,
        train_loader=trainLoader,
        test_loader=testLoader,
        device=device,
    )

    trainer.fit()


def loadModel(modelDir, nInputLayers=24, device="cuda"):
    modelWeightPath = os.path.join(modelDir, "modelWeights")
    with open(os.path.join(modelDir, "args"), "rb") as handle:
        cfg = pickle.load(handle)

    if not isinstance(cfg, DictConfig):
        raise TypeError(
            "Loaded config is not a DictConfig. "
            "Models must be trained with the new Hydra-based configuration."
        )

    # Exclude 'name' from being passed to the model constructor
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg.pop("name", None)
    model = hydra.utils.instantiate(
        model_cfg,
        nDays=nInputLayers,
        device=device,
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Hydra changes working directory to output_subdir by default.
    # We want to keep control or at least know where we are.
    # The config usually has outputDir.
    # trainModel expects outputDir to be the target.

    # If we use hydra to manage output dir, cfg.outputDir might be redundant or conflicting.
    # For now, we'll just respect the cfg.outputDir as the user intent.
    # trainModel ensures it exists.

    # However, to handle relative paths in datasetPath correctly if hydra changes cwd:
    # Hydra changes cwd to something like outputs/2023-xx-xx/...
    # datasetPath is relative to project root "./data/..."
    # We should use hydra.utils.get_original_cwd() to resolve paths if they are relative.

    orig_cwd = hydra.utils.get_original_cwd()

    if not os.path.isabs(cfg.datasetPath):
        cfg.datasetPath = os.path.join(orig_cwd, cfg.datasetPath)

    if not os.path.isabs(cfg.outputDir):
        cfg.outputDir = os.path.join(orig_cwd, cfg.outputDir)

    trainModel(cfg)


if __name__ == "__main__":
    main()
