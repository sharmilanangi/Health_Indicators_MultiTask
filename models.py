import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskNet(nn.Module):
    def __init__(self, layer_sizes=[462, 512, 256, 128]):
        super().__init__()

        self.mlp_net = nn.Sequential(
            nn.Linear(layer_sizes[0], layer_sizes[1]),  ## 96x64
            nn.ReLU(),
            nn.Linear(layer_sizes[1], layer_sizes[2]),  ## 64x1
            nn.ReLU(),
            nn.Linear(layer_sizes[2], layer_sizes[3]),
        )

        self.last_layer = nn.Linear(
            layer_sizes[3], 1
        )  ## change if we need classification or softmax

    def forward(self, x):
        x = self.mlp_net(x)
        out_x = self.last_layer(x)
        return out_x
