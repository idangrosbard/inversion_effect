from typing import List, Callable, Dict, OrderedDict
from torch import Tensor
from tqdm import tqdm
import pandas as pd



class LowRankModel(object):
    def __init__(self, metric: Callable[[Tensor], Dict[str, List[float]]]):
        self.metric = metric

    def __call__(self, state_dict: OrderedDict[str, Tensor]) -> OrderedDict[str, List[float | str | int]]:
        return self.low_rank_approx(state_dict)
    
    def low_rank_approx(self, w: OrderedDict[str, Tensor]) -> OrderedDict[str, List[float | str | int]]:
        """
        Low rank approximation of the matrix.
        """
        model_summary = {'layer': []}
        for key in tqdm(w, desc='Layer...'):
            if len(w[key].shape) > 2:
                w[key] = w[key].reshape(w[key].shape[0], -1)
            if len(w[key].shape) == 1:
                continue
            # Perform SVD
            w_spectral_dist = self.metric(w[key])
            for k in w_spectral_dist:
                n_rows = len(w_spectral_dist[k])
                if k not in model_summary:
                    model_summary[k] = w_spectral_dist[k]
                else:
                    model_summary[k].extend(w_spectral_dist[k])
            model_summary['layer'].extend([key] * n_rows)
            
        return pd.DataFrame(model_summary)
    
    