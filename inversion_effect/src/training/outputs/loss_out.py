from torch import Tensor
from ..datasets import SampleType
from .base_out import BaseOut

class LossOut(BaseOut):
    def __init__(self, loss: Tensor, n: int, type: SampleType | int):
        super(LossOut, self).__init__(n, type)
        self.loss = loss
        
        
        