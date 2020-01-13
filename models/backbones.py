import torch
from torch import nn
# import torch.nn.functional as F
try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


##################################################################################
# VGG16 network definition
##################################################################################
# class Vgg16(nn.Module):
#     def __init__(self):
#         super(Vgg16, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
#
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
#
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, X):
#         h = F.relu(self.conv1_1(X), inplace=True)
#         h = F.relu(self.conv1_2(h), inplace=True)
#         # relu1_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv2_1(h), inplace=True)
#         h = F.relu(self.conv2_2(h), inplace=True)
#         # relu2_2 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv3_1(h), inplace=True)
#         h = F.relu(self.conv3_2(h), inplace=True)
#         h = F.relu(self.conv3_3(h), inplace=True)
#         # relu3_3 = h
#         h = F.max_pool2d(h, kernel_size=2, stride=2)
#
#         h = F.relu(self.conv4_1(h), inplace=True)
#         h = F.relu(self.conv4_2(h), inplace=True)
#         h = F.relu(self.conv4_3(h), inplace=True)
#         # relu4_3 = h
#         conv4_4 = self.conv4_4(h)
#
#         h = F.relu(self.conv5_1(h), inplace=True)
#         h = F.relu(self.conv5_2(h), inplace=True)
#         h = F.relu(self.conv5_3(h), inplace=True)
#         relu5_3 = h
#
#         return relu5_3
#         # return [relu1_2, relu2_2, relu3_3, relu4_3]


##################################################################################
# VGG19 network definition
##################################################################################
class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights is not None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
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

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x
