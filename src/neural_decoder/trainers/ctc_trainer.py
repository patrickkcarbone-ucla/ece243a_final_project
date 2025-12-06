import os
import time
import csv
import pickle
from datetime import datetime
import numpy as np
import torch
from edit_distance import SequenceMatcher
from neural_decoder.trainers.base_trainer import BaseTrainer


class CTCTrainer(BaseTrainer):
    def __init__(
        self,
        cfg,
        model,
        optimizer,
        scheduler,
        augmenter,
        train_loader,
        test_loader,
        device,
    ):
        super().__init__(cfg)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.augmenter = augmenter
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        self.loss_ctc = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True
        )

        # Detect if using dual-region dataset (6 items) vs single-region (5 items)
        sample_batch = next(iter(train_loader))
        self.dual_region = len(sample_batch) == 6

    def _unpack_batch(self, batch):
        """
        Unpack batch from dataloader, handling both single and dual-region formats.

        Returns:
            X: Primary neural features (6v for dual-region, all for single-region)
            X_44: Area 44 features (only for dual-region, else None)
            y: Phoneme sequences
            X_len: Neural sequence lengths
            y_len: Phoneme sequence lengths
            dayIdx: Day indices
        """
        if self.dual_region:
            # DualRegionSpeechDataset: (x_6v, x_44, phone_seq, time_bins, phone_lens, day)
            X, X_44, y, X_len, y_len, dayIdx = batch
            X_44 = X_44.to(self.device)
        else:
            # SpeechDataset: (neural_feats, phone_seq, time_bins, phone_lens, day)
            X, y, X_len, y_len, dayIdx = batch
            X_44 = None

        X = X.to(self.device)
        y = y.to(self.device)
        X_len = X_len.to(self.device)
        y_len = y_len.to(self.device)
        dayIdx = dayIdx.to(self.device)

        return X, X_44, y, X_len, y_len, dayIdx

    def fit(self):
        for batch in range(self.cfg.nBatch):
            self.model.train()
            self.augmenter.train()

            batch_data = next(iter(self.train_loader))
            X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)

            # Apply augmentations (pass X_44 for dual-region augmentations)
            X = self.augmenter(X, X_44)

            # Compute prediction error
            pred = self.model.forward(X, dayIdx)

            loss = self.loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - self.model.kernelLen) / self.model.strideLen).to(
                    torch.int32
                ),
                y_len,
            )
            loss = torch.sum(loss)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Eval
            eval_freq = getattr(self.cfg, "evalFrequency", 100)
            if batch % eval_freq == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.augmenter.eval()  # Ensure no noise during eval

                    # Evaluate on TRAIN set (subset for speed)
                    train_loss, train_cer = self._evaluate_loader(
                        self.train_loader, max_batches=5
                    )

                    # Evaluate on TEST set (full)
                    test_loss, test_cer = self._evaluate_loader(
                        self.test_loader
                    )

                    # Use the parent methods!
                    self.log_metrics(
                        batch, train_loss, train_cer, test_loss, test_cer
                    )
                    self.save_checkpoint(self.model, test_cer)
                    self.save_stats(train_loss, train_cer, test_loss, test_cer)

    def _evaluate_loader(self, loader, max_batches=None):
        """Evaluate model on a data loader, returning avg loss and CER."""
        all_loss = []
        total_edit_distance = 0
        total_seq_length = 0

        for batch_idx, batch_data in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            X, X_44, y, X_len, y_len, dayIdx = self._unpack_batch(batch_data)
            # No augmentation during eval

            pred = self.model.forward(X, dayIdx)
            loss = self.loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - self.model.kernelLen) / self.model.strideLen).to(
                    torch.int32
                ),
                y_len,
            )
            loss = torch.sum(loss)
            all_loss.append(loss.cpu().detach().numpy())

            adjustedLens = (
                (X_len - self.model.kernelLen) / self.model.strideLen
            ).to(torch.int32)

            for iterIdx in range(pred.shape[0]):
                decodedSeq = torch.argmax(
                    pred[iterIdx, 0 : adjustedLens[iterIdx], :],
                    dim=-1,
                )
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

        n_batches = (
            min(len(loader), max_batches) if max_batches else len(loader)
        )
        avg_loss = np.sum(all_loss) / n_batches
        cer = (
            total_edit_distance / total_seq_length
            if total_seq_length > 0
            else 1.0
        )

        return avg_loss, cer
