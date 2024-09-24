import torch
from typing import Tuple, OrderedDict, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm


def low_rank_approx_mat(w: torch.Tensor, rank: int) -> Tuple[torch.Tensor, float]:
    """
    Low rank approximation of the matrix.
    """
    # Perform SVD
    u, s, vt = torch.linalg.svd(w)
    # Low rank approximation
    approx_matrix = torch.dot(u[:, :rank], torch.dot(torch.diag(s[:rank]), vt[:rank, :]))
    if rank == 1:
        u_r = u[:, :rank].unsqueeze(1)
        s_r = torch.diag(s[:rank])
        vt_r = vt[:rank, :]
        approx_matrix = torch.dot(u_r, torch.dot(s_r, vt_r))
    else:
        approx_matrix = torch.dot(u[:, :rank], torch.dot(torch.diag(s[:rank]), vt[:rank, :]))
    
    spectral_dist = torch.norm(s[rank:], p=2)
    return approx_matrix, spectral_dist


def low_rank_approx_dist(w: torch.Tensor, rank: int) -> float:
    """
    Low rank approximation of the matrix.
    """
    # Perform SVD
    _, s, _ = torch.linalg.svd(w)
    # Low rank approximation
    s = s.abs()
    s = torch.sort(s, descending=True).values
    
    spectral_dist = torch.norm(s[rank:], p=2)
    return spectral_dist


def low_rank_approx_error(w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
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


def effective_rank(w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
    """
    Effective rank of the matrix.
    """
    # Perform SVD
    _, s, _ = torch.linalg.svd(w)
    # Effective rank (The Low-Rank Simplicity Bias in Deep Networks: https://arxiv.org/pdf/2103.10427)
    s = s
    distribution = s / s.sum()
    entropy = -torch.sum(distribution * torch.log(distribution))

    return {'effective_rank': [entropy]}


def low_rank_approx_spectral_norm(w: torch.Tensor) -> OrderedDict[str, List[int | float]]:
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
    s = torch.abs(s)
    s = torch.sort(s, descending=True).values
    # s = s

    spectral_dist['rank'].extend(range(min(w.shape)))
    spectral_dist['spectral_dist'].extend(s.tolist())
    
    return spectral_dist

def low_rank_approx_trend(w: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, List[float | str | int]]:
    """
    Low rank approximation of the matrix.
    """
    
    spectral_dist = {'layer': [], 'spectral_dist': [], 'rank': []}
    for key in tqdm(w, desc='Layer...'):
        if len(w[key].shape) > 2:
            w[key] = w[key].reshape(w[key].shape[0], -1)
        if len(w[key].shape) == 1:
            continue
        # Perform SVD
        w_spectral_dist = low_rank_approx_error(w[key])
        spectral_dist['layer'].extend([key] * len(w_spectral_dist['spectral_dist']))
        spectral_dist['spectral_dist'].extend(w_spectral_dist['spectral_dist'])
        spectral_dist['rank'].extend(w_spectral_dist['rank'])
    return pd.DataFrame(spectral_dist)


def plot_spectral_dist(all_spectral_dist: pd.DataFrame) -> go.Figure:
    """
    Plot the spectral distance.
    """
    import numpy as np
    base = np.array([20,20,20])
    target = np.array([220, 0, 0])
    n_layers = all_spectral_dist['layer'].nunique()
    color_map = [base + (target-base)*i/n_layers for i in range(n_layers)]
    color_map = [f'rgb({int(color[0])},{int(color[1])},{int(color[2])})' for color in color_map]
    fig = px.line(all_spectral_dist, x='rank', y='spectral_dist', color='layer', title='Spectral Distance Trend', color_discrete_sequence=color_map)
    fig.update_layout(
        xaxis_title="Approximation Rank",
        yaxis_title="Approximation Error",
        template='plotly_white'
    )
    return fig
