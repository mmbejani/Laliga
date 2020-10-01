import torch
from torch import nn
from model.networks.auxiliary_network import AuxiliaryNetwork, AuxiliaryLayer


class Network(AuxiliaryNetwork):

    def __init__(self):
        super().__init__()
        self.padding1 = nn.ZeroPad2d(6)
        self.conv1_1 = nn.Conv2d(3, 16, 5)
        self.conv1_2 = nn.Conv2d(16, 16, 5)
        self.conv1_3 = nn.Conv2d(16, 16, 5)
        self.sh_conv1 = nn.Conv2d(3, 16, 1)
        self.norm1 = nn.BatchNorm2d(16)
        self.aux1 = AuxiliaryLayer(self, [self.conv1_1, self.conv1_2, self.conv1_3, self.sh_conv1], [16, 32, 32])

        self.conv2_1 = nn.Conv2d(16, 32, 5)
        self.conv2_2 = nn.Conv2d(32, 32, 5)
        self.conv2_3 = nn.Conv2d(32, 32, 5)
        self.sh_conv2 = nn.Conv2d(16, 32, 1)
        self.norm2 = nn.BatchNorm2d(32)
        self.aux2 = AuxiliaryLayer(self, [self.conv2_1, self.conv2_2, self.conv2_3, self.sh_conv2], [32, 16, 16])

        self.conv3_1 = nn.Conv2d(32, 64, 2)
        self.conv3_2 = nn.Conv2d(64, 64, 2)
        self.conv3_3 = nn.Conv2d(64, 64, 2)
        self.norm3 = nn.BatchNorm2d(64)
        self.aux3 = AuxiliaryLayer(self, [self.conv3_1, self.conv3_2, self.conv3_3], [64, 5, 5])

        self.conv4_1 = nn.Conv2d(64, 128, 2)
        self.conv4_2 = nn.Conv2d(128, 128, 1)
        self.conv4_3 = nn.Conv2d(128, 128, 1)
        self.norm4 = nn.BatchNorm2d(128)
        self.aux4 = AuxiliaryLayer(self, [self.conv4_1, self.conv4_2, self.conv4_3], [128, 1, 1])

        self.linear1 = nn.Linear(128, 1024)
        self.aux5 = AuxiliaryLayer(self, [self.linear1], [1024])
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        sh1 = torch.relu(self.sh_conv1(x))
        y = torch.relu(self.conv1_1(x))
        y = torch.relu(self.conv1_2(y))
        y = torch.relu(self.conv1_3(y))
        y = self.padding1(y)
        y = y + sh1
        y = self.norm1(y)
        y1 = self.aux1(y)
        y = self.pool(y)

        sh2 = torch.relu(self.sh_conv2(y))
        y = torch.relu(self.conv2_1(y))
        y = torch.relu(self.conv2_2(y))
        y = torch.relu(self.conv2_3(y))
        y = self.padding1(y)
        y = y + sh2
        y = self.norm2(y)
        y2 = self.aux2(y)
        y = self.pool(y)

        y = torch.relu(self.conv3_1(y))
        y = torch.relu(self.conv3_2(y))
        y = torch.relu(self.conv3_3(y))
        y = self.norm3(y)
        y3 = self.aux3(y)
        y = self.pool(y)

        y = torch.relu(self.conv4_1(y))
        y = torch.relu(self.conv4_2(y))
        y = torch.relu(self.conv4_3(y))
        y = self.norm4(y)
        y4 = self.aux4(y)

        y = y.view([-1, 128])
        y = torch.relu(self.linear1(y))
        y5 = self.aux5(y)
        y = torch.relu(self.linear2(y))
        y = torch.softmax(self.linear3(y), dim=-1)

        return [y1, y2, y3, y4, y5, y]
