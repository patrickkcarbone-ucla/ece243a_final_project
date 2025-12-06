# Neural Speech Decoder - ECE143A/243A Final Project

**Patrick Carbone and Azad Azeus**

Brain-to-text decoding from intracortical neural signals using Transformer architectures with multi-level coarticulation modeling.

## Quick Start

### Setup

```bash
# Create environment
uv venv -p 3.9
source .venv/bin/activate
uv pip install -e .

# Download data from https://datadryad.org/dataset/doi:10.5061/dryad.x69p8czpq
# Unzip and rename to 'data/'

# Build diphone vocab
python scripts/build_diphone_vocab.py --dataset data/ptDecoder_ctc

# Format datasets (run the notebook)
notebooks/formatCompetitionLogdDualData.ipynb
```

### Training Commands

```bash
# Quick test (2 epochs)
python scripts/train_model.py experiment=transformer/tcp/quick/test_2epochs
```
