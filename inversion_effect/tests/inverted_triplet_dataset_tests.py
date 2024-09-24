import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.datasets import InvertedTripletDataset, SampleType

from typing import List
from pathlib import Path
import pytest
from tests.utils import mock_dataset
import torch
from torch import Tensor, distributions
from torchvision import transforms
from torchvision.datasets import ImageFolder





@mock_dataset
def test_mixed_triplet_dataset_inheritence(root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = InvertedTripletDataset(root=root, transform=t, should_invert_distr=distributions.Bernoulli(1.0))



def mixed_triplet_test(pr: float, root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inv_ds = InvertedTripletDataset(root=root, should_invert_distr=distributions.Bernoulli(pr), transform=t)
    # assert that the sample type is correct
    
    return inv_ds[0][-1]

@mock_dataset
def test_sample_type(root: Path):
    assert mixed_triplet_test(0.0, root) == SampleType.TRIPLET_UPRIGHT.value
    assert mixed_triplet_test(1.0, root) == SampleType.TRIPLET_INVERTED.value


@mock_dataset
def get_dataloader_sample_type(pr: float, root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inv_ds = InvertedTripletDataset(root=root, should_invert_distr=distributions.Bernoulli(pr), transform=t)
    # assert that the sample type is correct
    dl = torch.utils.data.DataLoader(inv_ds, batch_size=1, shuffle=False)

    sample_types = []
    
    for b in dl:
        sample_types.append(b[-1][0])
    return sample_types


def assert_dl_sample_type(sample_types: List[Tensor], sample_type: SampleType):
    for st in sample_types:
        assert st == sample_type.value


def test_dl_sample_type():
    assert_dl_sample_type(get_dataloader_sample_type(0.0), SampleType.TRIPLET_UPRIGHT)
    assert_dl_sample_type(get_dataloader_sample_type(1.0), SampleType.TRIPLET_INVERTED)


@mock_dataset
def test_inversion(root: Path):
    flipper = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
    t = transforms.Compose([transforms.Resize((224, 224))])
    inv_ds = InvertedTripletDataset(root=root, should_invert_distr=distributions.Bernoulli(1.0), transform=t)
    up_ds = InvertedTripletDataset(root=root, should_invert_distr=distributions.Bernoulli(0.0), transform=t)
    for i in range(len(inv_ds)):
        a1, _, _, _ = inv_ds[i]
        a2, _, _, _ = up_ds[i]
        assert (a1 == flipper(a2))
