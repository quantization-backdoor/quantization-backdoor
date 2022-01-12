import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers_quantize import Conv2dQuantize, LinearQuantize
from torch.quantization import QuantStub, DeQuantStub


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG16'):
        super(VGG, self).__init__()
        self.vgg_cfg = cfg[vgg_name]
        self.features = self._make_layers(self.vgg_cfg)
        self.classifier = LinearQuantize(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [Conv2dQuantize(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def reset_quantize(self, quantize):
        self.classifier.reset_quantize(quantize)
        i = 0
        for x in self.vgg_cfg:
            if x == 'M':
                i += 1
            else:
                self.features[i].reset_quantize(quantize)
                i += 3
