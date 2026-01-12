import torch
from torch.nn import Module, ModuleList

import torch.nn.functional as F

# functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class MimicVideo(Module):
    def __init__(self):
        super().__init__()

    def forward(self, video):
        pass
