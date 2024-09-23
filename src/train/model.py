import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torchvision.models import resnet18

from src.common.const import CONFIG

_log = logging.getLogger(__name__)


class ERBaseModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def last_conv_layer(self) -> nn.Module:
        pass


class Net(ERBaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        # (batch_size, 32, 48, 48)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # (batch_size, 64, 24, 24)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # (batch_size, 128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        # Dropout layer with probability 0.5
        self.dropout = nn.Dropout(p=0.5)
        # (batch_size, 512)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        # (batch_size, output_size)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    @property
    def last_conv_layer(self) -> nn.Conv2d:
        return self.conv3


class ResNet(ERBaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = resnet18()
        # Change the first layer to accept grayscale images
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Change the last layer to output 6 classes
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    @property
    def last_conv_layer(self) -> nn.Sequential:
        return self.model.layer4


class ExpressionNet(ERBaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Sequential(nn.Linear(4608, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3))
        self.fc2 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5))
        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

    @property
    def last_conv_layer(self) -> nn.Sequential:
        return self.conv4


class XpressionNet(ERBaseModel):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.1),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc1 = nn.Sequential(nn.Linear(4608, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25))
        self.fc2 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5))
        self.out = nn.Linear(512, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # multiply all dimensions except the first together
        flat_tensor_size = np.prod(x.shape[1:])
        # Use tensor.view() to reshape a tensor
        x = x.view(-1, flat_tensor_size)

        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

    @property
    def last_conv_layer(self) -> nn.Sequential:
        return self.conv4


MODELS = {
    "ExpressionNet": ExpressionNet,
    "XpressionNet": XpressionNet,
    "ResNet": ResNet,
}


class ModelNotFoundError(ValueError):
    def __init__(self, model_name: str):
        super().__init__(f"Model `{model_name}` not found in {list(MODELS.keys())}")


def get_model_from_path(model_path: str) -> nn.Module:
    """Return the model class based on the model path."""
    filename = os.path.basename(model_path)
    model_name = filename.split("_")[0]
    model = MODELS.get(model_name)
    if model is None:
        raise ModelNotFoundError(model_name)
    num_classes = len(CONFIG.labels)
    return model(num_classes)


def load_model_from_path(model_path: str, model: nn.Module, to_eval: bool = True) -> None:
    """Load the trained model from the given path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = type(model).__name__
    _log.info("Loading model from %s as %s", model_path, model_name)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    _log.info("Model %s loaded successfully to device: %s.", model_name, device)
    if to_eval:
        model.eval()
