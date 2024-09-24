import torch
from torch import nn, Tensor
from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Iterable
from collections import namedtuple
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm
from .outputs import BaseOut
from .datasets import SampleType



class BaseTrainer(object):
    def __init__(self, **kwargs):
        self.optimizer = kwargs.get("optimizer", None)
        self.scheduler = kwargs.get("scheduler", None)
        self.train_loader = kwargs.get("train_loader", None)
        self.val_loader = kwargs.get("val_loader", None)
        self.criterion = kwargs.get("criterion", None)
        self.num_epochs = kwargs.get("num_epochs", None)
        self.device = kwargs.get("device", None)
        self.writer = kwargs.get("writer", None)
        self.model = kwargs.get("model", None)
        self.eval_freq = kwargs.get("eval_freq", None)
        self.silent_tqdm = kwargs.get("silent_tqdm", False)

        # self.upright_acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)
        # self.inverted_acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)
        # self.acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)

        # self.epoch_loss = MeanMetric().to(self.device)
        # self.epoch_acc = MeanMetric().to(self.device)
        # self.epoch_upright_acc = MeanMetric().to(self.device)
        # self.epoch_inverted_acc = MeanMetric().to(self.device)

        self.type_metrics = kwargs.get("type_metrics", None)
        self.total_steps = 0
    
    
    @staticmethod
    def _filter_type(t: Tensor, type: Tensor, filter_type: SampleType) -> Tensor | None:
        t = t[type == filter_type.value]
        return t
    
    @staticmethod
    def _filter_batch_type(batch: Iterable[Tensor], type: Tensor, filter_type: SampleType) -> Iterable[Tensor]:
        return [BaseTrainer._filter_type(t, type, filter_type) for t in batch]
    
    @staticmethod
    def _total_loss(outputs: Dict[SampleType, BaseOut]) -> Tensor:
        sum_loss = 0
        for key, out in outputs.items():
            sum_loss += out.loss
        return sum_loss
        

    def _model_batch(self, batch: Iterable[Tensor], sample_type: Tensor) -> Tuple[Dict[SampleType, BaseOut], Tensor]:
        raise NotImplementedError


    def _log_batch_metrics(self, train: bool, batch_outputs: Dict[SampleType, BaseOut]):
        for key, out in batch_outputs.items():
            if key in self.type_metrics:
                self.type_metrics[key].log_batch(out, train, self.total_steps)


    def _batch(self, batch: Iterable[Tensor], sample_type: Tensor, train: bool) -> Dict[SampleType, BaseOut]:
        if train:
            self.optimizer.zero_grad()
        for i in range(len(batch)):
            batch[i] = batch[i].to(self.device)
        sample_type = sample_type.to(self.device)
        
        all_outputs, loss = self._model_batch(batch, sample_type)
        
        
        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.total_steps)

        return all_outputs
    

    def _log_epoch_metrics(self, train: bool, epoch: int):
        for key in self.type_metrics.keys():
            self.type_metrics[key].log_epoch(epoch, train)

        
    def _epoch(self, train: bool):
        if train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        for batch in tqdm(loader, disable=self.silent_tqdm):
            sample_type = batch[-1]
            
            outputs = self._batch(batch[:-1], sample_type, train)

            self._log_batch_metrics(train, outputs)

            self.total_steps += 1


    def train(self):
        # Train loop
        for epoch in tqdm(range(self.num_epochs), disable=self.silent_tqdm):
            # train:
            print(f"Train epoch: {epoch}")
            self.model.train()
            self._epoch(train=True)
            self._log_epoch_metrics(train=True, epoch=epoch)

            # eval:
            if self.eval_freq and epoch % self.eval_freq == 0:
                with torch.no_grad():
                    print(f"Eval epoch: {epoch}")
                    self.model.eval()
                    self._epoch(train=False)
                    self._log_epoch_metrics(train=False, epoch=epoch)
                    self.writer.flush()

        with torch.no_grad():
            print(f"Test epoch:")
            self.model.eval()
            self._epoch(train=False)
            self._log_epoch_metrics(train=False, epoch=epoch)
            self.writer.flush()
        

