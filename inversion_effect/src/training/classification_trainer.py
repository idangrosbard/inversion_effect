from .base_trainer import BaseTrainer
from torch import Tensor
from typing import Iterable
from .datasets import SampleType
from .outputs import ClassificationOut, BaseOut


class ClassificationTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ClassificationTrainer, self).__init__(**kwargs)
    

    def _model_batch(self, batch: Iterable[Tensor], sample_type: SampleType) -> BaseOut:
        x, y = batch[0], batch[1]
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)

        return ClassificationOut(loss, len(y), y, y_hat, type=sample_type)
    
