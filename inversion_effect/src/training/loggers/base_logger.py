from torch import Tensor
import torch
from torchmetrics import MetricCollection, MeanMetric
from torch.utils.tensorboard import SummaryWriter
from ..outputs import BaseOut
from ..datasets import SampleType


class BaseLogger(object):
    def __init__(self, writer: SummaryWriter, type: SampleType | int, device: torch.device):
        self.epoch_loss = MeanMetric().to(device)
        self.writer = writer
        if isinstance(type, int):
            type = SampleType(type)
        self.type = type

    def log_batch(self, out: BaseOut, train: bool, step: int):
        assert out.type == self.type, f"Output type {out.type} does not match logger type {self.type}"
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/batch_{self.type}_loss", out.loss.item(), step)
        self.epoch_loss.update(out.loss, out.n)

    def log_epoch(self, epoch: int, train: bool):
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/batch_{self.type}_loss", self.epoch_loss.compute().item(), epoch)
        self.epoch_loss.reset()