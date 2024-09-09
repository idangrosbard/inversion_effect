# mixed faces
nohup python inversion_effect/scripts/train.py --train_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/train" --val_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/val" --batch_size 256 --num_epochs 10 --log_dir /home/new_storage/inversion_effect/mixed_faces --output_path /home/new_storage/inversion_effect/mixed_faces_model_weights.pth --num_epochs 10 --eval_freq 5 --silent_tqdm --inversion_pr 0.5

# upright faces
nohup python inversion_effect/scripts/train.py --train_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/train" --val_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/val" --batch_size 256 --num_epochs 10 --log_dir /home/new_storage/inversion_effect/upright_faces --output_path /home/new_storage/inversion_effect/upright_faces_model_weights.pth --num_epochs 10 --eval_freq 5 --silent_tqdm --inversion_pr 0.0

# mixed inanimates
nohup python inversion_effect/scripts/train.py --train_pth "/home/ssd_storage/datasets/processed/num_classes/260_inanimate_imagenet_num-classes_260/train" --val_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/val" --batch_size 256 --num_epochs 10 --log_dir /home/new_storage/inversion_effect/mixed_inanimates --output_path /home/new_storage/inversion_effect/mixed_inanimates_model_weights.pth --num_epochs 10 --eval_freq 5 --silent_tqdm --inversion_pr 0.5

# upright inanimates
nohup python inversion_effect/scripts/train.py --train_pth "//home/ssd_storage/datasets/processed/num_classes/260_inanimate_imagenet_num-classes_260/train" --val_pth "/home/ssd_storage/datasets/processed/230_faces_vggface_idx_num-classes_230_{'train': 0.8, 'val': 0.2}/val" --batch_size 256 --num_epochs 10 --log_dir /home/new_storage/inversion_effect/upright_inanimates --output_path /home/new_storage/inversion_effect/upright_inanimates_model_weights.pth --num_epochs 10 --eval_freq 5 --silent_tqdm --inversion_pr 0.0
