from torch import Tensor
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from .base_logger import BaseLogger
from ..outputs import ClassificationOut
from ..datasets import SampleType


class ClassificationLogger(BaseLogger):
    def __init__(self, writer: SummaryWriter, type: SampleType | int, num_classes: int, device: torch.device):
        super(ClassificationLogger, self).__init__(writer, type, device)
        self.batch_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.epoch_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    def log_batch(self, out: ClassificationOut, train: bool, step: int):
        super(ClassificationLogger, self).log_batch(out, train, step)
        self.batch_accuracy.update(out.y_hat, out.y)
        self.epoch_accuracy.update(out.y_hat, out.y)
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/batch_{self.type}_accuracy", self.batch_accuracy.compute().item(), step)
        self.batch_accuracy.reset()

    def log_epoch(self, epoch: int, train: bool):
        super(ClassificationLogger, self).log_epoch(epoch, train)
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/epoch_{self.type}_accuracy", self.epoch_accuracy.compute().item(), epoch)
        self.epoch_accuracy.reset()