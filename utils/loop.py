import os

from copy import deepcopy

import torch
from torch import Tensor
from math import log10
from .metrics import Metrics

from typing import Any, Callable
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model: Module, loader: DataLoader, optimizer: Optimizer, loss_function: Callable[[Tensor], Tensor]) -> Module:
    model.train()
    model = model.to(device)
    # Print variables
    n = len(loader)
    n_digits = int(log10(n))

    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        images, labels = [x.to(device) for x in batch]
        predictions = model(images)

        loss = loss_function(predictions.flatten(), labels.flatten())
        loss.backward()
        optimizer.step()

        print(f"\r[TRAIN] Iteration: [{str(i + 1).rjust(n_digits)} / {len(loader)}] | Loss: {loss.item()}", end="")
    print()

    return model

def validate(model: Module, loader: DataLoader) -> Metrics:
    model.eval()
    model = model.to(device)

    # Print variables
    n = len(loader)
    n_digits = int(log10(n))

    n_dataset = len(loader.dataset)
    predictions = torch.zeros(n_dataset).to(device)
    labels_all = torch.zeros(n_dataset).to(device)

    position = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, labels = [x.to(device) for x in batch]
            B = images.shape[0]
            predictions[position: position + B] = model(images).flatten()
            labels_all[position: position + B] = labels
            position += B
            print(f"\r[VALIDATION] Iteration: [{str(i + 1).rjust(n_digits)} / {len(loader)}]", end="")
    
    metrics = Metrics()
    metrics.set_values(predictions, labels_all)
    metrics.get_AUC_and_ROC()
    print(f" AUC: {metrics.AUC}")

    return metrics


def test(model: Module, loader: DataLoader) -> Metrics:
    model.eval()
    model = model.to(device)

    # Print variables
    n = len(loader)
    n_digits = int(log10(n))

    n_dataset = len(loader.dataset)
    predictions = torch.zeros(n_dataset).to(device)
    labels_all = torch.zeros(n_dataset).to(device)

    position = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            images, labels = [x.to(device) for x in batch]
            B = images.shape[0]
            predictions[position: position + B] = model(images).flatten()
            labels_all[position: position + B] = labels
            position += B
            print(f"\r[TESTING] Iteration: [{str(i + 1).rjust(n_digits)} / {len(loader)}]", end="")
    
    metrics = Metrics()
    metrics.set_values(predictions, labels_all)
    metrics.get_AUC_and_ROC()
    print(f" AUC: {metrics.AUC}")

    return metrics



def save(path: str, epoch: int, model: Module, metric: Metrics) -> None:
    torch.save(model.cpu().state_dict(), os.path.join(path, f"model_{epoch + 1}.pt"))
    torch.save(metric, os.path.join(path, f"metrics_{epoch + 1}.pt"))

def loop(model: Module, optimizer: Optimizer, loss_function: Callable[[Tensor], Tensor], save_path: str,
         train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
         n_epochs: int, scheduler: LRScheduler = None) -> Module:
    
    model = model.to(device)
    
    AUC_best = 0
    epoch_best = -1
    model_best = None
    for epoch in range(n_epochs):
        print(f"[INFO] Epoch {epoch + 1}")
        model = train(model, train_loader, optimizer, loss_function)
        if scheduler: scheduler.step()
        metric = validate(model, valid_loader)
        if metric.AUC > AUC_best:
            AUC_best = metric.AUC
            epoch_best = epoch
            model_best = deepcopy(model.cpu())

    metric_best = test(model_best, test_loader)
    save(save_path, epoch_best, model_best, metric_best)

    return model.cpu()
