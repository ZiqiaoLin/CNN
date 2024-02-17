from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)

        self.pooling = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pooling(x)
        x = f.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        output = f.log_softmax(x, dim=1)
        return output


def cal_correction(output, target):
    pred = torch.argmax(output,dim=1)

    #correct = pred.eq(target.data.view_as(output)).sum()
    correct = torch.eq(pred,target).sum()
    percentage = (correct / len(target)) * 100

    return percentage
