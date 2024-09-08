import torch
from torch import nn, Tensor
from torch.optim import optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List
from collections import namedtuple
from torchmetrics import Accuracy, MeanMetric
from tqdm import tqdm


class Trainer(object):
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

        self.upright_acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)
        self.inverted_acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)
        self.acc = Accuracy(task="multiclass", num_classes=kwargs.get("num_classes", None)).to(self.device)
        self.epoch_loss = MeanMetric().to(self.device)
        self.epoch_acc = MeanMetric().to(self.device)
        self.epoch_upright_acc = MeanMetric().to(self.device)
        self.epoch_inverted_acc = MeanMetric().to(self.device)
        self.total_steps = 0
        


    def _batch(self, x: Tensor, y: Tensor, is_inverted: Tensor, train: bool) -> Tuple:
        if train:
            self.optimizer.zero_grad()
        x, y, is_inverted = x.to(self.device), y.to(self.device), is_inverted.to(self.device)
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        if train:
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        batch_up_acc = self.upright_acc(y_hat[1 - is_inverted], y[1 - is_inverted])
        batch_inv_acc = self.inverted_acc(y_hat[1 - is_inverted], y[1 - is_inverted])
        batch_acc = self.acc(y_hat, y)

        # named tuple
        BatchOutput = namedtuple("BatchOutput", ["loss", "batch_up_acc", "batch_inv_acc", "batch_total_acc"])

        output = BatchOutput(float(loss.item()), batch_up_acc.item(), batch_inv_acc.item(), batch_acc.item())

        return output
    

    def _log_batch_metrics(self, train: bool, batch_output, n: int, n_upright: int, n_inverted: int):
        str_mode = "train" if train else "val"

        self.writer.add_scalar(f"{str_mode}/batch_loss", batch_output.loss, self.total_steps)
        self.writer.add_scalar(f"{str_mode}/batch_acc", batch_output.batch_total_acc, self.total_steps)
        self.writer.add_scalar(f"{str_mode}/batch_upright_acc", batch_output.batch_up_acc, self.total_steps)
        self.writer.add_scalar(f"{str_mode}/batch_inverted_acc", batch_output.batch_inv_acc, self.total_steps)
        self.writer.flush()
        
        self.epoch_loss(batch_output.loss, n)
        self.epoch_acc(batch_output.batch_total_acc, n)
        self.epoch_upright_acc(batch_output.batch_up_acc, n_upright)
        self.epoch_inverted_acc(batch_output.batch_inv_acc, n_inverted)


    def _log_epoch_metrics(self, train: bool, epoch: int):
        str_mode = "train" if train else "val"

        self.writer.add_scalar(f"{str_mode}/epoch_loss", self.epoch_loss.compute(), epoch)
        self.writer.add_scalar(f"{str_mode}/epoch_acc", self.epoch_acc.compute(), epoch)
        self.writer.add_scalar(f"{str_mode}/epoch_upright_acc", self.epoch_upright_acc.compute(), epoch)
        self.writer.add_scalar(f"{str_mode}/epoch_inverted_acc", self.epoch_inverted_acc.compute(), epoch)
        self.writer.flush()
        self.epoch_loss.reset()
        self.epoch_acc.reset()
        self.epoch_upright_acc.reset()
        self.epoch_inverted_acc.reset()

        
    def _epoch(self, train: bool):
        if train:
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        for x, y, is_inverted in tqdm(loader):
            
            n = x.size(0)
            n_upright = torch.sum(1 - is_inverted).item()
            n_inverted = torch.sum(is_inverted).item()
            output = self._batch(x, y, is_inverted, train)

            self._log_batch_metrics(train, output, n, n_upright, n_inverted)

            self.total_steps += 1


    def train(self):
        for epoch in tqdm(range(self.num_epochs)):
            # train:
            print(f"Train epoch: {epoch}")
            self.model.train()
            self._epoch(train=True)
            self._log_epoch_metrics(train=True)

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
        

