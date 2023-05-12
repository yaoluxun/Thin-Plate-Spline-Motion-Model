import torch
import torch.nn as nn
import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.dense_motion import DenseMotionNetwork

class FineTunedInpaintingNetwork(nn.Module):
    def __init__(self, inpainting_network):
        super(FineTunedInpaintingNetwork, self).__init__()

        # Load the pretrained inpainting network
        self.inpainting_network = inpainting_network

        # Add new layers to the network
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Pass input through the pretrained inpainting network
        x = self.inpainting_network(x)

        # Pass output through the newly added layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.tanh(x)

        return x