from .base_trainer import BaseTrainer
from torch import Tensor
from typing import Iterable, Dict, Tuple
from .datasets import SampleType
from .outputs import ClassificationOut, LossOut, BaseOut


class ClassificationTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ClassificationTrainer, self).__init__(**kwargs)
    
    def _model_batch(self, batch: Iterable[Tensor], sample_type: Tensor) -> Tuple[Dict[SampleType, BaseOut], Tensor]:
        x, y = batch[0], batch[1]
        y_hat = self.model(x)

        loss = self.criterion(y_hat, y)

        outputs = {SampleType.CLASSIFY_BOTH: LossOut(loss, len(y), SampleType.CLASSIFY_BOTH)}

        for type in [SampleType.CLASSIFY_UPRIGHT, SampleType.CLASSIFY_INVERTED]:
            y_filtered = self._filter_type(y, sample_type, type)
            y_hat_filtered = self._filter_type(y_hat, sample_type, type)
            outputs[type] = ClassificationOut(len(y_filtered), y_filtered, y_hat_filtered, type=type)
        
        return outputs, loss
    
