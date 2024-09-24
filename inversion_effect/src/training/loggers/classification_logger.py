from .base_logger import BaseLogger
from ..outputs import ClassificationOut
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, MeanMetric
from datasets import SampleType


class ClassificationLogger(BaseLogger):
    def __init__(self, writer: SummaryWriter, type: SampleType | int):
        super(ClassificationLogger, self).__init__(writer)
        self.accuracy = Accuracy()
        self.epoch_accuracy = Accuracy()

    def log_batch(self, out: ClassificationOut, train: bool, step: int):
        super(ClassificationLogger, self).log_batch(out, train, step)
        self.accuracy.update(out.y_hat, out.y)
        self.epoch_accuracy.update(out.y_hat, out.y)
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/batch_{self.type}_accuracy", self.accuracy.compute().item(), step)
        self.accuracy.reset()

    def log_epoch(self, epoch: int, train: bool):
        super(ClassificationLogger, self).log_epoch(epoch, train)
        mode = "train" if train else "val"
        self.writer.add_scalar(f"{mode}/epoch_{self.type}_accuracy", self.epoch_accuracy.compute().item(), epoch)
        self.epoch_accuracy.reset()