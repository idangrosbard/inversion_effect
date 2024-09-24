import pandas as pd
from torch.utils.data.dataset import ConcatDataset, Dataset
from .triplet_dataset import TripletDataset
from .sample_type import SampleType
from typing import Any, Tuple
from PIL import Image
from torchvision import transforms



class MixedTripletDataset(TripletDataset):
    def __init__(self, pos_same_img: bool, neg_same_id: bool, **kwargs):
        super(MixedTripletDataset, self).__init__(**kwargs)
        assert (pos_same_img, neg_same_id) != (False, True), 'Cannot have same positive and negative type of samples'
        self.pos_same_img = pos_same_img
        self.neg_same_id = neg_same_id
    
    def _sample_row(self, cls: str, idx: int, is_positive: bool = True) -> pd.Series:
        # Get a positive sample
        if is_positive:
            if self.pos_same_img:
                # return the same image as positive
                return self.df.iloc[idx]
            else:
                # return a different image from the same class
                return super()._sample_row(cls, idx, is_positive)
            
        # Get a negative sample
        else:
            if self.neg_same_id:
                # return an image from the same class
                return super()._sample_row(cls, idx, True)
            else:
                # return an image from a different class
                return super()._sample_row(cls, idx, False)
    
    @staticmethod
    def _flip(img: Image.Image) -> Image.Image:
        flipper = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        return flipper(img)

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:

        # Set the sample type
        if self.pos_same_img:
            sample_type = SampleType.TRIPLET_MIXED_POS_IMG
        else:
            sample_type = SampleType.TRIPLET_MIXED_POS_ID
        
        # load images
        anchor, positive, negative = super().__getitem__(idx)
        
        # Flip the positive and negative images
        # flipped_positive = positive.transpose(Image..FLIP_TOP_BOTTOM)
        # flipped_negative = negative.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        flipped_positive = MixedTripletDataset._flip(positive)
        flipped_negative = MixedTripletDataset._flip(negative)

        return anchor, flipped_positive, flipped_negative, sample_type.value
        
    
    
