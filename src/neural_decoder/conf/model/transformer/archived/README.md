# Archived TCP Model Configs

These are older/smaller TCP Transformer model configs.

## What's Here

- `tcp.yaml` - Base TCP model (384 hidden_dim, 5 layers)
- `tcp_larger.yaml` - Larger model (480 hidden_dim, 6 layers)
- `tcp_xl.yaml` - XL model (576 hidden_dim, 7 layers)
- `tcp_xxl.yaml` - XXL model (720 hidden_dim, 8 layers)

## Current Best

Use `tcp_xxxl.yaml` or `tcp_xxxxl.yaml` (in parent directory):

| Model | hidden_dim | layers | heads | ffn_dim |
|-------|------------|--------|-------|---------|
| tcp_xxxl | 896 | 10 | 8 | 3584 |
| **tcp_xxxxl** | 1120 | 12 | 10 | 4480 |

