from .low_rank_metric import LowRankMetric
from typing import List, OrderedDict
import torch


class LowRankApproxError(LowRankMetric):
    def __call__(self, w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
        """
        Low rank approximation of the matrix.
        """
        spectral_dist = {'rank': [], 'spectral_dist': []}
        if len(w.shape) > 2:
                w = w.reshape(w.shape[0], -1)
        
        # for i in tqdm(range(min(w.shape)), desc='Rank...'):
        # Perform SVD
        # spectral_dist_r = low_rank_approx_dist(w, i)
        _, s, _ = torch.linalg.svd(w)
        # Low rank approximation
        s = s.abs()
        s = torch.sort(s, descending=True).values
        s = s ** 2
        s = s.cumsum(dim=0)
        s = s ** 0.5
        # s = s.flip(dims=(0,))
        s = s[-1] - s

        spectral_dist['rank'].extend(range(min(w.shape)))
        spectral_dist['spectral_dist'].extend(s.tolist())
            
        return spectral_dist