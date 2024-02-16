from typing import Union, Callable, Tuple, Any, Optional, Dict

import torch
from torch import nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class CNN(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,input):
        output = input + 1
        return output


model = CNN()
x = torch.tensor(1.0)
output = model(x)
print(output)