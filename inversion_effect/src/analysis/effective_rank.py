from .low_rank_metric import LowRankMetric
from typing import List, OrderedDict
import torch


class EffectiveRank(LowRankMetric):
    def __call__(self, w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
        return self.effective_rank(w)
    
    def effective_rank(self, w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
        """
        Effective rank of the matrix.
        """
        # Perform SVD
        _, s, _ = torch.linalg.svd(w)
        # Effective rank (The Low-Rank Simplicity Bias in Deep Networks: https://arxiv.org/pdf/2103.10427)
        s = s
        distribution = s / s.sum()
        entropy = -torch.sum(distribution * torch.log(distribution))

        return {'effective_rank': [entropy.item()]}