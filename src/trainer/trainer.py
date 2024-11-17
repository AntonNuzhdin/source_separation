import torch
from torch import Tensor
import random
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.metrics import SISNRi, SISDRi, PESQ, SDRi, STOI



class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config=config, *args, **kwargs)

        pesq_config = next((conf for conf in config.metrics.inference if conf["name"] == "PESQ"), None)
        stoi_config = next((conf for conf in config.metrics.inference if conf["name"] == "STOI"), None)

        self.sisnri = SISNRi()
        self.sisdr = SISDRi()

        if pesq_config:
            self.pesq = PESQ(fs=pesq_config["fs"], mode=pesq_config["mode"])

        if stoi_config:
            self.stoi = STOI(fs=stoi_config["fs"], extended=stoi_config.get("extended", False))

        self.sdri = SDRi()
        self.scaler = GradScaler()

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)

        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step(batch["loss"])

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode='train'):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode != "train":
            self._log_predictions(**batch)
            self._log_input_audio(**batch)
        else:
            self._log_input_audio(**batch)

    def _log_input_audio(self, s1_audio: Tensor, s2_audio: Tensor, mix_audio: Tensor, **batch):
        cnt = min(len(s1_audio), len(s2_audio), len(mix_audio))
        i = random.randint(0, cnt - 1)
        self.writer.add_audio("Audio1", s1_audio[i], 16000)
        self.writer.add_audio("Audio2", s2_audio[i], 16000)
        self.writer.add_audio("Mix_audio", mix_audio[i], 16000)

    def _log_predictions(self, s1_audio: Tensor, s2_audio: Tensor, speaker_1: Tensor,
                         speaker_2: Tensor, mix_audio: Tensor, **batch):
        cnt = min(len(s1_audio), len(s2_audio), len(speaker_1), len(speaker_2), len(mix_audio))
        i = random.randint(0, cnt - 1)
        self.writer.add_audio("Audio1", s1_audio[i], 16000)
        self.writer.add_audio("Predicted1", speaker_1[i], 16000)

        self.writer.add_audio("Audio2", s2_audio[i], 16000)
        self.writer.add_audio("Predicted2", speaker_2[i], 16000)

        self.writer.add_audio("Mix_audio", mix_audio[i], 16000)

        # self.writer.add_scalar("SISNRi", self.sisnri(
        #     s1_audio[i], s2_audio[i], mix_audio[i], speaker_1[i], speaker_2[i]
        # ))
        # self.writer.add_scalar("SISDRi", self.sisdr(
        #     s1_audio[i], s2_audio[i], mix_audio[i], speaker_1[i], speaker_2[i]
        # ))
        # self.writer.add_scalar("PESQ", self.pesq(
        #     s1_audio[i], s2_audio[i], mix_audio[i], speaker_1[i], speaker_2[i]
        # ))
        # self.writer.add_scalar("SDRi", self.sdri(
        #     s1_audio[i], s2_audio[i], mix_audio[i], speaker_1[i], speaker_2[i]
        # ))
        # self.writer.add_scalar("STOI", self.stoi(
        #     s1_audio[i], s2_audio[i], mix_audio[i], speaker_1[i], speaker_2[i]
        # ))
