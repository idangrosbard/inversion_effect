import torch
from torchvision.datasets import ImageFolder
from torch.distributions import Distribution
from typing import Tuple


class InversionDataset(ImageFolder):
    """
    Dataset for inversion.
    """

    def __init__(self, should_invert_distr: Distribution, **kwargs):
        super(InversionDataset, self).__init__(**kwargs)
        self.should_invert_distr = should_invert_distr
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        """
        Get the item.
        """
        img, cls = super(InversionDataset, self).__getitem__(idx)
        is_inverted = 0
        # Invert the image
        if self.should_invert_distr.sample().item():
            img = torch.flip(img, dims=[2])
            is_inverted = 1
        return img, cls, is_inverted
