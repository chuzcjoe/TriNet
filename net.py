# -*- coding:utf-8 -*-
"""
    Implementation of Pose Estimation with mobileNetV2
"""
import torch.nn as nn
import torch
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # building classifier
        self.fc_x1 = nn.Linear(self.last_channel, num_classes)
        self.fc_y1 = nn.Linear(self.last_channel, num_classes)
        self.fc_z1 = nn.Linear(self.last_channel, num_classes)

        self.fc_x2 = nn.Linear(self.last_channel, num_classes)
        self.fc_y2 = nn.Linear(self.last_channel, num_classes)
        self.fc_z2 = nn.Linear(self.last_channel, num_classes)

        self.fc_x3 = nn.Linear(self.last_channel, num_classes)
        self.fc_y3 = nn.Linear(self.last_channel, num_classes)
        self.fc_z3 = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x, phase='train'):
        x = self.features(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x_v1 = self.fc_x1(x)
        y_v1 = self.fc_y1(x)
        z_v1 = self.fc_z1(x)

        x_v2 = self.fc_x2(x)
        y_v2 = self.fc_y2(x)
        z_v2 = self.fc_z2(x)

        x_v3 = self.fc_x3(x)
        y_v3 = self.fc_y3(x)
        z_v3 = self.fc_z3(x)

        return x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3
    
    
"""
    Implementation of Head Pose Estimation with mobileNetV2
"""

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG16(nn.Module):
    def __init__(self, features, num_classes=66):
       super(VGG16, self).__init__()
       self.features = features
       self.last_channel = 1000
       
       # building classifier
       self.fc_x1 = nn.Linear(self.last_channel, num_classes)
       self.fc_y1 = nn.Linear(self.last_channel, num_classes)
       self.fc_z1 = nn.Linear(self.last_channel, num_classes)

       self.fc_x2 = nn.Linear(self.last_channel, num_classes)
       self.fc_y2 = nn.Linear(self.last_channel, num_classes)
       self.fc_z2 = nn.Linear(self.last_channel, num_classes)

       self.fc_x3 = nn.Linear(self.last_channel, num_classes)
       self.fc_y3 = nn.Linear(self.last_channel, num_classes)
       self.fc_z3 = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)

        x_v1 = self.fc_x1(x)
        y_v1 = self.fc_y1(x)
        z_v1 = self.fc_z1(x)

        x_v2 = self.fc_x2(x)
        y_v2 = self.fc_y2(x)
        z_v2 = self.fc_z2(x)

        x_v3 = self.fc_x3(x)
        y_v3 = self.fc_y3(x)
        z_v3 = self.fc_z3(x)

        return x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3

class ResNet(nn.Module):
    def __init__(self, features, num_classes):
       super(ResNet, self).__init__()

       self.features = features
       self.last_channel = 1000

       # building classifier
       self.fc_x1 = nn.Linear(self.last_channel, num_classes)
       self.fc_y1 = nn.Linear(self.last_channel, num_classes)
       self.fc_z1 = nn.Linear(self.last_channel, num_classes)

       self.fc_x2 = nn.Linear(self.last_channel, num_classes)
       self.fc_y2 = nn.Linear(self.last_channel, num_classes)
       self.fc_z2 = nn.Linear(self.last_channel, num_classes)

       self.fc_x3 = nn.Linear(self.last_channel, num_classes)
       self.fc_y3 = nn.Linear(self.last_channel, num_classes)
       self.fc_z3 = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)

        x_v1 = self.fc_x1(x)
        y_v1 = self.fc_y1(x)
        z_v1 = self.fc_z1(x)

        x_v2 = self.fc_x2(x)
        y_v2 = self.fc_y2(x)
        z_v2 = self.fc_z2(x)

        x_v3 = self.fc_x3(x)
        y_v3 = self.fc_y3(x)
        z_v3 = self.fc_z3(x)

        return x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3


class Densenet(nn.Module):
    def __init__(self, features, num_classes=66):
       super(Densenet, self).__init__()

       self.features = features
       self.last_channel = 1000

       # building classifier
       self.fc_x1 = nn.Linear(self.last_channel, num_classes)
       self.fc_y1 = nn.Linear(self.last_channel, num_classes)
       self.fc_z1 = nn.Linear(self.last_channel, num_classes)

       self.fc_x2 = nn.Linear(self.last_channel, num_classes)
       self.fc_y2 = nn.Linear(self.last_channel, num_classes)
       self.fc_z2 = nn.Linear(self.last_channel, num_classes)

       self.fc_x3 = nn.Linear(self.last_channel, num_classes)
       self.fc_y3 = nn.Linear(self.last_channel, num_classes)
       self.fc_z3 = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)

        x_v1 = self.fc_x1(x)
        y_v1 = self.fc_y1(x)
        z_v1 = self.fc_z1(x)

        x_v2 = self.fc_x2(x)
        y_v2 = self.fc_y2(x)
        z_v2 = self.fc_z2(x)

        x_v3 = self.fc_x3(x)
        y_v3 = self.fc_y3(x)
        z_v3 = self.fc_z3(x)

        return x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3

class VGG19(nn.Module):

    def __init__(self, features, pretrained= False, num_classes=66, init_weights=False):
        super(VGG19, self).__init__()
        #if pretrained:
        self.features = features
        #else:
        #    self.features = features
        #    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        #    self.classifier = nn.Sequential(
        #    nn.Linear(512 * 7 * 7, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 4096),
        #    nn.ReLU(True),
        #    nn.Dropout(),
        #    nn.Linear(4096, 1000),
        #)
        
        self.last_channel = 1000
        
        # building classifier
        self.fc_x1 = nn.Linear(self.last_channel, num_classes)
        self.fc_y1 = nn.Linear(self.last_channel, num_classes)
        self.fc_z1 = nn.Linear(self.last_channel, num_classes)

        self.fc_x2 = nn.Linear(self.last_channel, num_classes)
        self.fc_y2 = nn.Linear(self.last_channel, num_classes)
        self.fc_z2 = nn.Linear(self.last_channel, num_classes)

        self.fc_x3 = nn.Linear(self.last_channel, num_classes)
        self.fc_y3 = nn.Linear(self.last_channel, num_classes)
        self.fc_z3 = nn.Linear(self.last_channel, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)
        
        x_v1 = self.fc_x1(x)
        y_v1 = self.fc_y1(x)
        z_v1 = self.fc_z1(x)

        x_v2 = self.fc_x2(x)
        y_v2 = self.fc_y2(x)
        z_v2 = self.fc_z2(x)

        x_v3 = self.fc_x3(x)
        y_v3 = self.fc_y3(x)
        z_v3 = self.fc_z3(x)

        return x_v1, y_v1, z_v1, x_v2, y_v2, z_v2, x_v3, y_v3, z_v3

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg19_bn(pretrained=False, progress=True, **kwargs):
    
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)




