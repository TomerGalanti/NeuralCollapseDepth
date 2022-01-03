"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn
from models import modules

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, settings, features):
        super().__init__()
        self.features = features
        self.top_layers_type = settings.top_layers_type
        self.top_depth = settings.top_depth
        self.num_input_channels = settings.num_input_channels
        self.num_output_classes = settings.num_output_classes
        self.relu = nn.ReLU()

        self.bottom_layers = nn.Sequential(
            nn.Linear(512, 4096),
            self.relu,
            nn.BatchNorm1d(4096),
            nn.Linear(4096, 4096),
            self.relu,
            nn.BatchNorm1d(4096)
        )

        self.fc = nn.Linear(4096, 128)

        if self.top_layers_type == 'fc':
            self.top_layers = nn.ModuleList([nn.Linear(128, 128) for i in range(self.top_depth)])
        else:
            self.top_layers = nn.ModuleList([modules.FCBlock(128, 50) for i in range(self.top_depth)])

        self.top_linear = nn.Linear(128, self.num_output_classes)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.bottom_layers(output)

        emb = self.fc(output)
        embeddings = [emb]
        output = self.relu(emb)

        # standard ResNet if self.top_depth = 0
        for i in range(self.top_depth):
            emb = self.top_layers[i](output)
            output = self.relu(emb)
            embeddings += [emb]

        output = self.top_linear(output)

        return output, embeddings

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn(settings):
    return VGG(settings, make_layers(cfg['A'], batch_norm=True))

def vgg13_bn(settings):
    return VGG(settings, make_layers(cfg['B'], batch_norm=True))

def vgg16_bn(settings):
    return VGG(settings, make_layers(cfg['D'], batch_norm=True))

def vgg19_bn(settings):
    return VGG(settings, make_layers(cfg['E'], batch_norm=True))