from torch import Tensor
import torch
from torchmetrics import MetricCollection, MeanMetric
from torch.utils.tensorboard import SummaryWriter
from ..outputs import BaseOut
from ..datasets import SampleType


class BaseLogger(object):
    def __init__(self, writer: SummaryWriter, type: SampleType | int, device: torch.device):
        self.writer = writer
        if isinstance(type, int):
            type = SampleType(type)
        self.type = type

    def log_batch(self, out: BaseOut, train: bool, step: int):
        raise NotImplementedError

    def log_epoch(self, epoch: int, train: bool):
        raise NotImplementedError