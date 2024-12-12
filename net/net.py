

import torch
from torchvision import models as torch_models
from net import sewer_models
from torch import nn


class ClassifierNet(nn.Module):

    TORCHVISION_MODEL_NAMES = sorted(name for name in torch_models.__dict__ if
                                     name.islower() and not name.startswith("__") and callable(
                                         torch_models.__dict__[name]))
    SEWER_MODEL_NAMES = sorted(name for name in sewer_models.__dict__ if
                               name.islower() and not name.startswith("__") and callable(sewer_models.__dict__[name]))
    MODEL_NAMES = TORCHVISION_MODEL_NAMES + SEWER_MODEL_NAMES

    def __init__(self, net_type='resnet18', num_classes=10):
        super(ClassifierNet, self).__init__()
        if net_type in ClassifierNet.TORCHVISION_MODEL_NAMES:
            self.model = torch_models.__dict__[net_type](num_classes = num_classes)
        elif net_type in ClassifierNet.SEWER_MODEL_NAMES:
            self.model = sewer_models.__dict__[net_type](num_classes = num_classes)
        else:
            raise ValueError("Got model {}, but no such model is in this codebase".format(net_type))

    def forward(self, x):
        out = self.model(x)
        return out


if __name__ == '__main__':
    net=ClassifierNet('resnet18')
    x=torch.randn(1,3,125,125)
    print(net(x).shape)
