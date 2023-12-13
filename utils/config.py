import json
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from typing import Any, Callable
import torch
from torch import Tensor
import torch.nn.functional as F

class Config:
    def __init__(self, params_dict: dict[str, Any]) -> None:
        # Basic hyperparameters
        self.learning_rate: float = None
        self.batch_size: int = None
        self.n_epochs: int = None

        # Data
        self.folder_real: str = None
        self.folders: list[str] = None
        self.folders_train: list[str] = None
        self.folders_valid: list[str] = None
        self.folders_both: list[str] = None
        self.folders_test: list[str] = None

        # Model
        self.model_type: str = None
        self.model_weights: str = None

        # Train modules
        self.loss_function: Callable[[Tensor], Tensor] = None
        self.optimizer: Optimizer = None
        self.scheduler: LRScheduler = None

        # Other
        self.time: str = datetime.now().strftime("%d-%m-%Y, %H:%M")
        self.name: str = None
        self.other: dict[str, Any] = dict()


        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__dict__[key] = value
            else:
                self.__dict__["other"][key] = value


    def save(self, path: str) -> None:
        np.save(path, self, allow_pickle=True)
        
        json_dict = dict()
        for key, value in self.__dict__.items():
            if value is not None and key in ["loss_function", "optimizer", "scheduler"]:
                value = value.__name__
            json_dict[key] = value

        with open(f"{path}.json", "w") as f:
            json.dump(json_dict, f, indent=4)
            

    def __str__(self) -> None:
        out = ""
        for key, value in self.__dict__.items():
            if value is not None and key in ["loss_function", "optimizer", "scheduler"]:
                value = value.__name__
            out += f"{key}: {value}\n"
        return out[:-1]

if __name__ == "__main__":
    D = {
        "learning_rate": 1e-4,
        "batch_size": 64,
        "n_epochs": 20,
        "folders": ["faces/real/"],
        "folders_train": ["faces/fake/FaceSwap", "faces/fake/Deepfakes"],
        "folders_valid": ["faces/fake/NeuralTextures"],
        "model_type": "resnet50",
        "model_weights_path": "ResNet50_Weights.IMAGENET1K_V2",
        "loss_function": F.binary_cross_entropy_with_logits,
        "optimizer": torch.optim.Adam,
        "random_parameter": [1, 2, "a", "b", None]
    }

    config = Config(D)

    print(config)