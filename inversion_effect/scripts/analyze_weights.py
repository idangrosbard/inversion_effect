import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import rank_analysis
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_pth", type=Path)
    parser.add_argument("--ref_model_pth", type=Path, default=None)
    parser.add_argument("--output_pth", type=Path, default='./spectral_dist.html')
    args = parser.parse_args()

    model = torch.load(args.model_pth, map_location='cpu')
    
    all_spectral_dist = rank_analysis.low_rank_approx_trend(model)

    if args.ref_model_pth is not None:
        ref_model = torch.load(args.ref_model_pth, map_location='cpu')
        ref_spectral_dist = rank_analysis.low_rank_approx_trend(ref_model)
        
        joined = pd.merge(all_spectral_dist, ref_spectral_dist, left_on=['rank','layer'], right_on=['rank','layer'], suffixes=('_new', '_ref'))
        
        joined['spectral_dist'] = joined['spectral_dist_new'] - joined['spectral_dist_ref']
        
        all_spectral_dist = joined[['layer', 'rank', 'spectral_dist']]

    fig = rank_analysis.plot_spectral_dist(all_spectral_dist)
    fig.write_html(args.output_pth)


if __name__ == "__main__":
    main()