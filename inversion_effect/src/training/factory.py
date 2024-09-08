from torch import nn, optim, distributions
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from .inversion_dataset import InversionDataset
from pathlib import Path


class Factory(object):
    def get_model(n_classes: int, model_arch: str = 'vgg16') -> nn.Module:
        """
        Get the model.
        """
        model = models.__dict__[model_arch](num_classes=n_classes)
        return model

    def get_optimizer(model: nn.Module, lr: float) -> optim.Optimizer:
        """
        Get the optimizer.
        """
        return optim.AdamW(model.parameters(), lr=lr)

    def get_scheduler(optimizer: optim.Optimizer, num_epochs: int, train_loader: DataLoader, max_lr: float = 0.001) -> optim.lr_scheduler._LRScheduler:
        """
        Get the scheduler.
        """
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_epochs * len(train_loader))
    
    def get_writer(log_dir: str) -> SummaryWriter:
        """
        Get the writer.
        """
        return SummaryWriter(log_dir=log_dir)
    
    def get_criterion() -> nn.Module:
        """
        Get the criterion.
        """
        return nn.CrossEntropyLoss()
    
    def get_data_loader(train_pth: Path, batch_size: int, flip_prob = 0.5, num_workers: int = 4, train: bool = True) -> DataLoader:
        """
        Get the train/test loader.
        """
        if train:
            tt = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        else:
            tt = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        ds = InversionDataset(
            root=train_pth, 
            transform=tt, should_invert_distr=distributions.Bernoulli(flip_prob))
        
        return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
