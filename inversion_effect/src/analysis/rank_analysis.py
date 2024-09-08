import torch
from typing import Tuple, OrderedDict, List
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def low_rank_approx_mat(w: torch.Tensor, rank: int) -> Tuple[torch.Tensor, float]:
    """
    Low rank approximation of the matrix.
    """
    # Perform SVD
    u, s, vt = torch.linalg.svd(w)
    # Low rank approximation
    approx_matrix = torch.dot(u[:, :rank], torch.dot(torch.diag(s[:rank]), vt[:rank, :]))
    spectral_dist = torch.norm(s[rank:], p=2)
    return approx_matrix, spectral_dist


def low_rank_approx_model(w: OrderedDict[str, torch.Tensor], rank: int) -> Tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, List[float | str]]]:
    """
    Low rank approximation of the matrix.
    """
    approx_matrix = {}
    spectral_dist = {'layer': [], 'spectral_dist': []}
    for key in w:
        # Perform SVD
        u, s, vt = torch.svd(w[key])
        # Low rank approximation
        approx_matrix[key] = torch.dot(u[:, :rank], torch.dot(torch.diag(s[:rank]), vt[:rank, :]))
        spectral_dist['layer'].append(key)
        spectral_dist['spectral_dist'].append(torch.norm(s[rank:], p=2).item())
    return approx_matrix, spectral_dist


def get_low_rank_approx_distance_trend(w: OrderedDict[str, torch.Tensor]):
    """
    Get the low rank approximation distance trend.
    """
    all_spectral_dist = []
    for i in range(1, 101):
        _, spectral_dist = low_rank_approx_model(w, i)
        all_spectral_dist.append(pd.DataFrame(spectral_dist))
        all_spectral_dist[-1]['rank'] = i
    
    all_spectral_dist = pd.concat(all_spectral_dist)

    return all_spectral_dist


def plot_spectral_dist(all_spectral_dist: pd.DataFrame) -> go.Figure:
    """
    Plot the spectral distance.
    """
    fig = px.line(all_spectral_dist, x='rank', y='spectral_dist', color='layer', title='Spectral Distance Trend')
    return fig

