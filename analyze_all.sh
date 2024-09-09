# fixed faces
python inversion_effect/scripts/analyze_weights.py --model_pth /home/new_storage/inversion_effect/mixed_faces_model_weights.pth --output_pth /home/new_storage/inversion_effect/mixed_faces_low_rank_approx.html

# upright inanimates
python inversion_effect/scripts/analyze_weights.py --model_pth /home/new_storage/inversion_effect/upright_inanimates_model_weights.pth --output_pth /home/new_storage/inversion_effect/upright_inanimates_low_rank_approx.html

# mixed inanimates
python inversion_effect/scripts/analyze_weights.py --model_pth /home/new_storage/inversion_effect/mixed_inanimates_model_weights.pth --output_pth /home/new_storage/inversion_effect/mixed_inanimates_low_rank_approx.html



# compare inannimates:
python inversion_effect/scripts/analyze_weights.py --model_pth /home/new_storage/inversion_effect/mixed_inanimates_model_weights.pth --ref_model_pth /home/new_storage/inversion_effect/upright_inanimates_model_weights.pth --output_pth /home/new_storage/inversion_effect/inanimates_mixed_upright_rank_comparison.html

# compare faces:
python inversion_effect/scripts/analyze_weights.py --model_pth /home/new_storage/inversion_effect/mixed_faces_model_weights.pth --ref_model_pth /home/new_storage/inversion_effect/upright_faces_model_weights.pth --output_pth /home/new_storage/inversion_effect/faces_mixed_upright_rank_comparison.html