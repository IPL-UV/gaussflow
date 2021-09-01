import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Sequential):
    def __init__(
        self, in_features, out_features, hidden_features, activation=nn.ReLU()
    ):
        layers = []

        for in_features, out_features in zip(
            [in_features] + hidden_features[:-1], hidden_features
        ):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation)

        # last layer is fully connected
        layers.append(nn.Linear(hidden_features[-1], out_features))
        super(MLP, self).__init__(*layers)
