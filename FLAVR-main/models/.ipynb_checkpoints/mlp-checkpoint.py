import torch.nn as nn
import torch

from models import register
from pdb import set_trace as bp

@register('mlp')
class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        coord_dim = 10
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        # x = self.layers[1](self.layers[0](x.view(-1, x.shape[-1])))
        # x = torch.cat([x, coord.view(-1, coord.shape[-1])], dim=1)
        # x = self.layers[3](self.layers[2](x))
        # x = torch.cat([x, coord.view(-1, coord.shape[-1])], dim=1)
        # x = self.layers[4](x)
        return x.view(*shape, -1)
