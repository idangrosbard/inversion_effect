import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from .inversion_dataset import InversionDataset
from src.training import InversionDataset


from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch


def test_inversion_dataset_inheritence():
    # create a mock dataset
    root = Path("./tmp/mock_dataset")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        cls_dir = root / str(i)
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, size=(32, 32, 3)).astype(np.uint8))
        img.save(cls_dir / f"{i}.png")
        
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = InversionDataset(root=root, transform=t, should_invert_distr=torch.distributions.Bernoulli(1.0))
    
    # cleanup
    for i in range(10):
        cls_dir = root / str(i)
        for img_path in cls_dir.iterdir():
            img_path.unlink()
        cls_dir.rmdir()


def inversion_test(pr: float):
    # create a mock dataset
    root = Path("./tmp/mock_dataset")
    root.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        cls_dir = root / str(i)
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, size=(32, 32, 3)).astype(np.uint8))
        img.save(cls_dir / f"{i}.png")
        
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = InversionDataset(root=root, transform=t, should_invert_distr=torch.distributions.Bernoulli(pr))
    true_imgs = []
    inversion_imgs = []
    for i in range(10):
        true_imgs.append(real_ds[i][0])
        inversion_imgs.append(inv_ds[i][0])
        
    # cleanup
    for i in range(10):
        cls_dir = root / str(i)
        for img_path in cls_dir.iterdir():
            img_path.unlink()
        cls_dir.rmdir()
    return true_imgs, inversion_imgs

def test_inversion_dataset_should_invert():
    true_imgs, inversion_imgs = inversion_test(1.0)
    for i in range(10):
        assert (true_imgs[i] == torch.flip(inversion_imgs[i], [2])).all()
        
    

def test_inversion_dataset_shouldnt_invert():
    true_imgs, inversion_imgs = inversion_test(0.0)
    for i in range(10):
        assert (true_imgs[i] == inversion_imgs[i]).all()
    