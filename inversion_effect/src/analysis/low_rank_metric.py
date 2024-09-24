from torch import Tensor
from typing import Any, List, Callable, Dict, OrderedDict


class LowRankMetric(Callable):
    """
    Abstract class for low rank metric calculation
    """
    def __call__(self, w: Tensor) -> Any:
        raise NotImplementedError