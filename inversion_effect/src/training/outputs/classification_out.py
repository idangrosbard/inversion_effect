from .base_out import BaseOut
from torch import Tensor
from ..datasets import SampleType


class ClassificationOut(BaseOut):
    def __init__(self, n: int, y: Tensor, y_hat: Tensor, type: SampleType | int):
        super(ClassificationOut, self).__init__(n, type)
        self.y = y
        self.y_hat = y_hat
        
