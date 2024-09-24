import torch
from torchvision.datasets import ImageFolder
from torch.distributions import Distribution
from typing import Tuple
from .sample_type import SampleType
from PIL import Image
from torchvision import transforms


class InvertedClassificationDataset(ImageFolder):
    """
    Dataset for inversion.
    """

    def __init__(self, should_invert_distr: Distribution, **kwargs):
        super(InvertedClassificationDataset, self).__init__(**kwargs)
        self.should_invert_distr = should_invert_distr

    @staticmethod
    def _flip(img: Image.Image | torch.Tensor) -> Image.Image | torch.Tensor:
        flipper = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        return flipper(img)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        """
        Get the item.
        """
        img, cls = super(InvertedClassificationDataset, self).__getitem__(idx)
        is_inverted = SampleType.CLASSIFY_UPRIGHT
        # Invert the image
        if self.should_invert_distr.sample().item():
            img = InvertedClassificationDataset._flip(img)
            is_inverted = SampleType.CLASSIFY_INVERTED
        return img, cls, is_inverted.value
