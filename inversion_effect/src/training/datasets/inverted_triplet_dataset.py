import torch
from torch.distributions import Distribution
from typing import Tuple
from .triplet_dataset import TripletDataset
from .sample_type import SampleType
from PIL import Image
from torchvision import transforms


class InvertedTripletDataset(TripletDataset):
    """
    Dataset for inversion.
    """

    def __init__(self, should_invert_distr: Distribution, **kwargs):
        super(InvertedTripletDataset, self).__init__(**kwargs)
        self.should_invert_distr = should_invert_distr


    def _flip(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        flipper = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        return flipper(img)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get the item.
        """
        a, p, n = super(InvertedTripletDataset, self).__getitem__(idx)
        is_inverted = SampleType.TRIPLET_UPRIGHT
        # Invert the image
        if self.should_invert_distr.sample().item():
            a = self._flip(a)
            p = self._flip(p)
            n = self._flip(n)
            is_inverted = SampleType.TRIPLET_INVERTED
        return a, p, n, is_inverted.value
