import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_trend(all_spectral_dist: pd.DataFrame, x_axis: str = 'rank', y_axis: str = 'spectral_dist') -> go.Figure:
    """
    Plot the spectral distance.
    """
    import numpy as np
    base = np.array([20,20,20])
    target = np.array([220, 0, 0])
    n_layers = all_spectral_dist['layer'].nunique()
    color_map = [base + (target-base)*i/n_layers for i in range(n_layers)]
    color_map = [f'rgb({int(color[0])},{int(color[1])},{int(color[2])})' for color in color_map]
    fig = px.line(all_spectral_dist, x=x_axis, y=y_axis, color='layer', title='Spectral Distance Trend', color_discrete_sequence=color_map)
    fig.update_layout(
        xaxis_title="Approximation Rank",
        yaxis_title="Approximation Error",
        template='plotly_white'
    )
    return fig


def plot_bar(all_spectral_dist: pd.DataFrame, x_axis: str = 'layer', y_axis: str = 'entropy') -> go.Figure:
    """
    Plot the spectral distance.
    """
    import numpy as np
    base = np.array([20,20,20])
    target = np.array([220, 0, 0])
    n_layers = all_spectral_dist['layer'].nunique()
    color_map = [base + (target-base)*i/n_layers for i in range(n_layers)]
    color_map = [f'rgb({int(color[0])},{int(color[1])},{int(color[2])})' for color in color_map]
    fig = px.bar(all_spectral_dist, x=x_axis, y=y_axis, color=x_axis, title='Layers effective rank', color_discrete_sequence=color_map)
    fig.update_layout(
        xaxis_title="Layers",
        yaxis_title="Effective Rank (Shannon Entropy)",
        template='plotly_white'
    )
    return fig