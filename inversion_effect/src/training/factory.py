from torch import nn, optim, distributions
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from pathlib import Path
from typing import Dict
import numpy as np
from .datasets.inversion_dataset import InvertedClassificationDataset
from .datasets import SampleType
from .loggers import BaseLogger, ClassificationLogger



class Factory(object):
    def get_model(n_classes: int, model_arch: str = 'vgg16') -> nn.Module:
        """
        Get the model.
        """
        model = models.__dict__[model_arch](num_classes=n_classes)
        return model

    def get_optimizer(model: nn.Module, lr: float, use_adam: bool = True) -> optim.Optimizer:
        """
        Get the optimizer.
        """
        if use_adam:
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    

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
    
    def get_data_loader(train_pth: Path, batch_size: int, flip_prob = 0.5, num_workers: int = 4, train: bool = True, imagenet: bool = False) -> DataLoader:
        """
        Get the train/test loader.
        """
        if imagenet:
            train_tt = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomCrop((224, 224)),
                transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            test_tt = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:

            # def pil_to_tensor(x):
                # return torch.tensor(np.array(x)).permute(2, 0, 1).float() / 255.0
                

            train_tt = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                # to_np,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            test_tt = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                # to_np,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        if train:
            tt = train_tt
        else:
            tt = test_tt

        ds = InvertedClassificationDataset(
            root=train_pth, 
            transform=tt, should_invert_distr=distributions.Bernoulli(flip_prob))
        
        return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers)
    
    def get_loggers(training_type: str, writer: SummaryWriter, num_classes: int | None, device: torch.device) -> Dict[SampleType, BaseLogger]:
        """
        Get the logger.
        """
        assert training_type in ['classification', 'triplet']
        
        loggers = {}
        if training_type == 'classification':
            for key in [SampleType.CLASSIFY_UPRIGHT, SampleType.CLASSIFY_INVERTED]:
                loggers[key] = ClassificationLogger(writer, key, num_classes, device)
            return loggers
        else:
            raise NotImplementedError
        
