import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """Standard dataset for single-region (256 channel) data."""
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )


class DualRegionSpeechDataset(Dataset):
    """
    Dataset for dual-region (512 channel) data that returns 6v and 44 separately.
    
    Used for Area44-based augmentation strategies where we need access to both
    regions during training (e.g., CrossAreaMixup, AdaptiveNoise44).
    
    Channel layout (512 total):
        - Channels 0-255: Area 6v (tx1 + spikePow from ventral premotor)
        - Channels 256-511: Area 44 (tx1 + spikePow from Broca's area)
    
    Returns:
        x_6v: [T, 256] Area 6v features
        x_44: [T, 256] Area 44 features  
        phone_seq: [max_seq] phoneme sequence
        time_bins: scalar, number of time bins
        phone_lens: scalar, length of phoneme sequence
        day: scalar, day index
    """
    
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)
        
        # Validate that data has 512 channels
        if len(self.neural_feats) > 0 and self.neural_feats[0].shape[1] != 512:
            raise ValueError(
                f"DualRegionSpeechDataset expects 512 channels, "
                f"got {self.neural_feats[0].shape[1]}. "
                "Use SpeechDataset for single-region data."
            )

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)
        
        # Split into Area 6v (0-255) and Area 44 (256-511)
        x_6v = neural_feats[:, :256]
        x_44 = neural_feats[:, 256:]

        if self.transform:
            # Transform only applies to 6v (the signal we decode from)
            x_6v = self.transform(x_6v)

        return (
            x_6v,
            x_44,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
