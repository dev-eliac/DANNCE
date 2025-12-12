from torchvision import models
import torch
import torch.nn as nn
import torch.nn.init as init
"""
ResNet adpated from Matsuura et al. 2020
(Code) https://github.com/mil-tokyo/dg_mmld/
(Paper) https://arxiv.org/pdf/1911.07661.pdf 
"""


def base_resnet(num_classes, arch='resnet18'):
    backbone = models.__dict__[arch](pretrained=True)
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Linear(num_ftrs, num_classes)
    nn.init.xavier_uniform_(backbone.fc.weight, .1)
    nn.init.constant_(backbone.fc.bias, 0.)
    return backbone


class Resnet(nn.Module):
    def __init__(self, num_classes, arch='resnet18'):

        super(Resnet, self).__init__()
        base_model = base_resnet(num_classes=num_classes, arch=arch)
        self.base_model = base_model

        self.conv_features = torch.nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu,
            base_model.maxpool, base_model.layer1, base_model.layer2,
            base_model.layer3, base_model.layer4, base_model.avgpool,
            torch.nn.Flatten(start_dim=1))
        # No dense features really
        self.dense_features = torch.nn.Identity()

        self.features = torch.nn.Sequential(self.conv_features,
                                            self.dense_features)

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        output_class = self.base_model.fc(x)
        return output_class

    def classifier(self, x):
        x = self.base_model.fc(x)
        return x