import os
import csv
import pickle
import time
import numpy as np
import torch
from datetime import datetime


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.startTime = time.time()
        self.testLoss = []
        self.testCER = []
        self.trainLoss = []
        self.trainCER = []

    def log_metrics(
        self, batch_idx, train_loss, train_cer, test_loss, test_cer
    ):
        endTime = time.time()
        log_msg = (
            f"batch {batch_idx}, "
            f"train_loss: {train_loss:>7f}, train_cer: {train_cer:>7f}, "
            f"test_loss: {test_loss:>7f}, test_cer: {test_cer:>7f}, "
            f"time/batch: {(endTime - self.startTime)/100:>7.3f}"
        )
        print(log_msg)

        # Text file logging
        try:
            timestamp = datetime.now().isoformat(timespec="seconds")
            with open(
                os.path.join(self.cfg.outputDir, "training_log.txt"), "a"
            ) as f:
                f.write(f"{timestamp} - {log_msg}\n")
        except Exception as e:
            print(f"Warning: log write failed: {e}")

        self.startTime = time.time()

    def save_checkpoint(self, model, current_cer):
        # Check if weight saving is enabled (default: False)
        save_weights = getattr(self.cfg, "saveModelWeights", False)
        if not save_weights:
            return

        # Only save if this is the best model so far
        if len(self.testCER) > 0 and current_cer < np.min(self.testCER):
            torch.save(
                model.state_dict(),
                os.path.join(self.cfg.outputDir, "modelWeights"),
            )

    def save_stats(self, train_loss, train_cer, test_loss, test_cer):
        self.trainLoss.append(train_loss)
        self.trainCER.append(train_cer)
        self.testLoss.append(test_loss)
        self.testCER.append(test_cer)

        tStats = {
            "trainLoss": np.array(self.trainLoss),
            "trainCER": np.array(self.trainCER),
            "testLoss": np.array(self.testLoss),
            "testCER": np.array(self.testCER),
        }

        # 1. Pickle (Legacy)
        with open(
            os.path.join(self.cfg.outputDir, "trainingStats"), "wb"
        ) as file:
            pickle.dump(tStats, file)

        # 2. CSV (Readable)
        try:
            with open(
                os.path.join(self.cfg.outputDir, "trainingStats.csv"),
                "w",
                newline="",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "eval_idx",
                        "trainLoss",
                        "trainCER",
                        "testLoss",
                        "testCER",
                    ]
                )
                for idx, (trl, trc, tel, tec) in enumerate(
                    zip(
                        self.trainLoss,
                        self.trainCER,
                        self.testLoss,
                        self.testCER,
                    )
                ):
                    writer.writerow([idx, trl, trc, tel, tec])
        except Exception as e:
            print(f"Warning: CSV save failed: {e}")

    def fit(self):
        raise NotImplementedError("Subclasses must implement fit()")
