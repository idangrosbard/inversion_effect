from torch import Tensor
from ..datasets import SampleType

class BaseOut(object):
    def __init__(self, loss: Tensor, n: int, type: SampleType | int):
        self.loss = loss
        self.n = n
        if isinstance(type, int):
            type = SampleType(type)
        self.type = type
        
        