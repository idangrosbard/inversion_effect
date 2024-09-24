from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from typing import Optional, Callable, Any, Tuple
from pathlib import Path
import pandas as pd


class TripletDataset(Dataset):
    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        allow_empty: bool = False,
    ):
        self.root = root
        self.transform = transform
        self.allow_empty = allow_empty
        cls_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        self.cls = cls_dirs
        cls_imgs = {d: list(d.iterdir()) for d in cls_dirs}
        for k in cls_imgs.keys():
            cls_imgs[k] = [img for img in cls_imgs[k] if img.suffix in IMG_EXTENSIONS]
        
        df = {'class': [], 'img': []}
        for k, v in cls_imgs.items():
            for img in v:
                df['class'].append(k)
                df['img'].append(img)
        self.df = pd.DataFrame(df)
        
    
    def __len__(self):
        return len(self.df)
    
    def _sample_row(self, cls: str, idx: int, is_positive: bool = True) -> pd.Series:
        condition = self.df['class'] == cls
        if is_positive:
            condition = condition
        else:
            condition = ~condition
        
        subset = self.df[condition]
        row = subset.sample(1).iloc[0]
        return row
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        # Load the anchor
        row = self.df.iloc[idx]
        cls = row['class']
        anchor = default_loader(row['img'])
        
        # Load the positive
        positive_row = self._sample_row(cls, idx, is_positive=True)
        positive = default_loader(positive_row['img'])

        # Load the negative
        negative_row = self._sample_row(cls, idx, is_positive=False)
        negative = default_loader(negative_row['img'])

        # Apply the transform
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative




