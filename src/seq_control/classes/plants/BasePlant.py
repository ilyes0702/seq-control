from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from seq_control.utils.plotting_utils import *


class BasePlant(ABC):
    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def control_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, u: torch.Tensor, t: float, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        pass