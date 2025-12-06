from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neural_decoder.neural_decoder_trainer import main as hydra_main


if __name__ == "__main__":
    """
    Thin wrapper around the Hydra-powered trainer.

    Usage is now identical to calling:
        python -m neural_decoder.neural_decoder_trainer

    You can override config options from the CLI, e.g.:
        python scripts/train_model.py outputDir=./logs/speech_logs/exp1
        python scripts/train_model.py model=baseline_gru optimizer.lr=0.01
    """
    hydra_main()
