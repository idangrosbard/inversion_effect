import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import rank_analysis
from pathlib import Path
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_pth", type=Path)
    parser.add_argument("--output_pth", type=Path, default='./spectral_dist.html')
    args = parser.parse_args()

    model = torch.load(args.model_pth)
    all_spectral_dist = rank_analysis.get_low_rank_approx_distance_trend(model)
    fig = rank_analysis.plot_spectral_dist(all_spectral_dist)
    fig.write_html(args.output_pth)
