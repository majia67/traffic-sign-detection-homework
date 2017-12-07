import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

nclasses = 43 # GTSRB as 43 classes

def Net():
    model = resnet34()
    model.fc = nn.Linear(512, nclasses)
    return model
