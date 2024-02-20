from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        output = f.log_softmax(x, dim=1)
        return output


def cal_correction(output, target):
    pred = torch.argmax(output, dim=1)

    # correct = pred.eq(target.data.view_as(output)).sum()
    correct = torch.eq(pred, target).sum()
    percentage = (correct / len(target)) * 100

    return percentage

if __name__ == '__main__':
    model = ConvNet()
    print(model)
    input = torch.ones((64,1,28,28))
    output = model(input)
    print(output.shape)