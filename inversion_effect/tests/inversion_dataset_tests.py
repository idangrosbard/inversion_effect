import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from .inversion_dataset import InversionDataset
from src.training.datasets import InvertedClassificationDataset, SampleType
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from typing import List, Tuple
from tests.utils import mock_dataset, N_CLS, N_IMGS




@mock_dataset
def test_inversion_dataset_inheritence(root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = InvertedClassificationDataset(root=root, transform=t, should_invert_distr=torch.distributions.Bernoulli(1.0))


@mock_dataset
def inversion_test(pr: float, root: Path) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = InvertedClassificationDataset(root=root, transform=t, should_invert_distr=torch.distributions.Bernoulli(pr))
    true_imgs = []
    inversion_imgs = []
    for i in range(N_CLS * N_IMGS):
        true_imgs.append(real_ds[i][0])
        inversion_imgs.append(inv_ds[i][0])
    return true_imgs, inversion_imgs



def assert_ds_sample_type(root: Path, pr: float, sample_type: SampleType):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inv_ds = InvertedClassificationDataset(root=root, transform=t, should_invert_distr=torch.distributions.Bernoulli(pr))
    # assert that the sample type is correct
    for i in range(N_CLS * N_IMGS):
        assert inv_ds[i][-1] == sample_type.value

@mock_dataset
def test_ds_sample_type(root: Path):
    assert_ds_sample_type(root, 1.0, SampleType.CLASSIFY_INVERTED)
    assert_ds_sample_type(root, 0.0, SampleType.CLASSIFY_UPRIGHT)


def test_inversion_dataset_should_invert():
    true_imgs, inversion_imgs = inversion_test(1.0)
    for i in range(N_CLS * N_IMGS):
        assert (true_imgs[i] == torch.flip(inversion_imgs[i], [2])).all()
    

def test_inversion_dataset_shouldnt_invert():
    true_imgs, inversion_imgs = inversion_test(0.0)
    for i in range(N_CLS * N_IMGS):
        assert (true_imgs[i] == inversion_imgs[i]).all()
    