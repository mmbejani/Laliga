import torch
from torch import nn
from typing import List, Iterable
import numpy as np
from copy import copy


class PassiveAuxiliaryLoss(nn.Module):

    def __init__(self, net: nn.Module, loss_function):
        super().__init__()
        self.net = net
        self.loss_function = loss_function

    def forward(self, x, y):
        outputs = self.net(x)
        losses = list()
        for output in outputs:
            losses.append(self.loss_function(output, y))
        return losses


class AuxiliaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_weights = list()
        self.o_weights = list()
        self.r_weights = list()
        self.t_param = list()


class AuxiliaryLayer(nn.Module):

    def __init__(self, net: AuxiliaryNetwork, layers: List[nn.Module], output_shape: Iterable, num_classes=10):
        super().__init__()
        weight_list = list()
        for l in layers:
            if isinstance(l, nn.Sequential):
                weight_list = self.add_weights_sequential(l, weight_list)
            else:
                weight_list.append(l.weight)
        self.input_shape = copy(output_shape)
        self.linear = nn.Linear(int(np.prod(output_shape)), num_classes)
        weight_list.append(self.linear.weight)
        net.r_weights.append(weight_list)
        if len(net.c_weights) > 0:
            net.c_weights.append(net.c_weights[-1][:-1] + weight_list)
        else:
            net.c_weights.append(weight_list)
        net.o_weights.append(net.c_weights[-1][:-1])

    def forward(self, x):
        x = x.view([-1, int(np.prod(self.input_shape))])
        x = torch.softmax(self.linear(x), dim=-1)
        return x

    def get_param(self):
        return list(self.linear.parameters())

    @staticmethod
    def add_weights_sequential(seq: nn.Sequential, weight_list: List):
        for l in seq:
            if isinstance(l, nn.Conv2d):
                weight_list.append(l.weight)
        return weight_list
