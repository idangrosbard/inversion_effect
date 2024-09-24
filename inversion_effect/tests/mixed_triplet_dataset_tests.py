import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from .inversion_dataset import InversionDataset
from src.training.datasets import MixedTripletDataset, SampleType
from pathlib import Path
from tests.utils import mock_dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytest
import torch
from torch import Tensor
from typing import List



@mock_dataset
def test_mixed_triplet_dataset_inheritence(root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    real_ds = ImageFolder(root=root, transform=t)
    inv_ds = MixedTripletDataset(root=root, pos_same_img=True, neg_same_id=False, transform=t)


@mock_dataset
def mixed_triplet_test(pos_same_img: bool, neg_same_id: bool, root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inv_ds = MixedTripletDataset(root=root, pos_same_img=pos_same_img, neg_same_id=neg_same_id, transform=t)
    # assert that the sample type is correct
    
    return inv_ds[0][-1]


def test_sample_type():
    assert mixed_triplet_test(True, True) == SampleType.TRIPLET_MIXED_POS_IMG.value
    assert mixed_triplet_test(True, False) == SampleType.TRIPLET_MIXED_POS_IMG.value
    assert mixed_triplet_test(False, False) == SampleType.TRIPLET_MIXED_POS_ID.value
    pytest.raises(AssertionError, mixed_triplet_test, False, True)


@mock_dataset
def get_dataloader_sample_type(pos_same_img: bool, neg_same_id: bool, root: Path):
    t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    inv_ds = MixedTripletDataset(root=root, pos_same_img=pos_same_img, neg_same_id=neg_same_id, transform=t)
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
    assert_dl_sample_type(get_dataloader_sample_type(True, True), SampleType.TRIPLET_MIXED_POS_IMG)
    assert_dl_sample_type(get_dataloader_sample_type(True, False), SampleType.TRIPLET_MIXED_POS_IMG)
    assert_dl_sample_type(get_dataloader_sample_type(False, False), SampleType.TRIPLET_MIXED_POS_ID)


@mock_dataset
def test_same_img_pos(root: Path):
    t = transforms.Compose([transforms.Resize((224, 224))])
    inv_ds = MixedTripletDataset(root=root, pos_same_img=True, neg_same_id=True, transform=t)
    anchor = inv_ds[0][0]
    pos = inv_ds[0][1]
    neg = inv_ds[0][2]
    assert (anchor == MixedTripletDataset._flip(pos))
    assert (anchor != MixedTripletDataset._flip(neg))
    assert (pos != neg)


@mock_dataset
def test_diff_img_pos(root: Path):
    t = transforms.Compose([transforms.Resize((224, 224))])
    inv_ds = MixedTripletDataset(root=root, pos_same_img=False, neg_same_id=False, transform=t)
    anchor = inv_ds[0][0]
    pos = inv_ds[0][1]
    neg = inv_ds[0][2]
    
    assert (anchor != MixedTripletDataset._flip(pos))
    assert (anchor != MixedTripletDataset._flip(neg))
    assert (pos != neg)