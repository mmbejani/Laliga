from model.networks.auxiliary_network import AuxiliaryLayer, AuxiliaryNetwork
import torch.nn as nn
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

num_class = 10
class VGG(AuxiliaryNetwork):
    def __init__(self, vgg_name='VGG11'):
        super().__init__()
        self.features = list()
        self.aux = list()
        last_shape = [3, 32, 32]
        for x in cfg[vgg_name]:
            feature = self._make_layer(x, last_shape[0])
            if x == 'M':
                self.features.append(['M', feature])
                last_shape[1] /= 2
                last_shape[2] /= 2
            else:
                self.features.append(['C', feature])
                self.t_param += list(feature.parameters())
                last_shape = self.compute_conv2d_output_shape(last_shape, [x, 3, 3])
                self.aux.append(AuxiliaryLayer(self, [feature], last_shape, num_class).cuda())
                self.t_param += self.aux[-1].get_param()
        self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, num_class)
        self.t_param += list(self.classifier.parameters())

    def forward(self, x):
        y = list()
        aux_counter = 0
        for f in self.features:
            x = f[1](x)
            if f[0] == 'C':
                y.append(self.aux[aux_counter](x))
                aux_counter += 1

        out = self.avg_pool(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return y + [out]

    def _make_layer(self, cfg, in_channels=3):
        if cfg == 'M':
            return nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2)]).cuda()
        return nn.Sequential(*[nn.Conv2d(in_channels, cfg, kernel_size=3, padding=1),
                               nn.Dropout(0.2),###################################################
                               nn.BatchNorm2d(cfg),
                               nn.ReLU(inplace=True)]).cuda()

    @staticmethod
    def compute_conv2d_output_shape(input_shape, kernel=[16, 3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1]):
        w_in = input_shape[1]
        h_in = input_shape[2]
        w_out = np.floor((w_in + 2 * padding[0] - dilation[0] * (kernel[1] - 1) - 1) / stride[0] + 1)
        h_out = np.floor((h_in + 2 * padding[1] - dilation[1] * (kernel[2] - 1) - 1) / stride[1] + 1)

        return [kernel[0], w_out, h_out]
