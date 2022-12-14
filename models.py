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


class MultiTaskNet(nn.Module):
    def __init__(
        self, shared_layer_sizes=[462, 512, 256], task_head_layers=[256, 128, 64, 1]
    ):
        super().__init__()

        self.mlp_net = nn.Sequential(
            nn.Linear(shared_layer_sizes[0], shared_layer_sizes[1]),  ## 462 x 512
            nn.ReLU(),
            nn.Linear(shared_layer_sizes[1], shared_layer_sizes[2]),  ## 512 x 256
        )

        self.last_layer_bmi = nn.Sequential(
            nn.Linear(task_head_layers[0], task_head_layers[1]),  ## 462 x 512
            nn.ReLU(),
            nn.Linear(task_head_layers[1], task_head_layers[2]),  ## 512
            nn.ReLU(),
            nn.Linear(task_head_layers[2], task_head_layers[3]),
        )

        # self.last_layer_cmr = nn.Sequential(
        #     nn.Linear(task_head_layers[0], task_head_layers[1]),  ## 462 x 512
        #     nn.ReLU(),
        #     nn.Linear(task_head_layers[1], task_head_layers[2]),  ## 512
        #     nn.ReLU(),
        #     nn.Linear(task_head_layers[2], task_head_layers[3]),
        # )
        self.last_layer_cmr = nn.Sequential(
            nn.Linear(task_head_layers[0], task_head_layers[3]),
        )

    def forward(self, x):
        x = self.mlp_net(x)

        out_bmi = self.last_layer_bmi(x)
        out_cmr = self.last_layer_cmr(x)

        return out_bmi, out_cmr
