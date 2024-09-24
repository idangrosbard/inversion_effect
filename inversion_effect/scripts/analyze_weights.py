import sys
import os
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.analysis import rank_analysis
from src.analysis import LowRankModel, EffectiveRank, LowRankApproxError, plot_trend, plot_bar
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_pth", type=Path)
    parser.add_argument("--ref_model_pth", type=Path, default=None)
    parser.add_argument("--output_pth", type=Path, default='./spectral_dist.html')
    args = parser.parse_args()

    model = torch.load(args.model_pth, map_location='cpu', weights_only=True)

    approx_analysis = LowRankModel(LowRankApproxError())
    effective_rank_analysis = LowRankModel(EffectiveRank())
    
    all_spectral_dist = approx_analysis(model)

    if args.ref_model_pth is not None:
        ref_model = torch.load(args.ref_model_pth, map_location='cpu', weights_only=True)
        ref_spectral_dist = approx_analysis(ref_model)
        
        joined = pd.merge(all_spectral_dist, ref_spectral_dist, left_on=['rank','layer'], right_on=['rank','layer'], suffixes=('_new', '_ref'))
        
        joined['spectral_dist'] = joined['spectral_dist_new'] - joined['spectral_dist_ref']
        
        all_spectral_dist = joined[['layer', 'rank', 'spectral_dist']]
    else:
        effective_rank = effective_rank_analysis(model)
        effective_rank_output_pth = args.output_pth.parent / (str(args.output_pth.stem) + '_effective_rank.html')
        plot_bar(effective_rank, y_axis='effective_rank').write_html(effective_rank_output_pth)

    fig = plot_trend(all_spectral_dist)
    fig.write_html(args.output_pth)
    

if __name__ == "__main__":
    main()
