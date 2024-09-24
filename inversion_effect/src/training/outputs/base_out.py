from torch import Tensor
from ..datasets import SampleType

class BaseOut(object):
    def __init__(self, n: int, type: SampleType | int):
        self.n = n
        if isinstance(type, int):
            type = SampleType(type)
        self.type = type
        
        