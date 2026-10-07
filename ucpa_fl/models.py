from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class MNISTCNN(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,3,padding=1); self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,3,padding=1); self.bn2=nn.BatchNorm2d(64)
        self.pool=nn.MaxPool2d(2); self.drop1=nn.Dropout(0.25)
        self.fc1=nn.Linear(64*14*14,128); self.drop2=nn.Dropout(0.5); self.fc2=nn.Linear(128,n_classes)
    def forward(self,x,return_features=False):
        x=F.relu(self.bn1(self.conv1(x))); x=self.pool(F.relu(self.bn2(self.conv2(x)))); x=self.drop1(x)
        x=x.flatten(1); f=F.relu(self.fc1(x)); x=self.drop2(f); y=self.fc2(x)
        return (y,f) if return_features else y

class CIFARSmallCNN(nn.Module):
    def __init__(self,n_classes=10):
        super().__init__()
        self.c1=nn.Conv2d(3,64,3,padding=1); self.b1=nn.BatchNorm2d(64)
        self.c2=nn.Conv2d(64,64,3,padding=1); self.b2=nn.BatchNorm2d(64)
        self.c3=nn.Conv2d(64,128,3,padding=1); self.b3=nn.BatchNorm2d(128)
        self.c4=nn.Conv2d(128,128,3,padding=1); self.b4=nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(2); self.drop=nn.Dropout(0.5)
        self.fc1=nn.Linear(128*8*8,256); self.fc2=nn.Linear(256,n_classes)
    def forward(self,x,return_features=False):
        x=F.relu(self.b1(self.c1(x))); x=F.relu(self.b2(self.c2(x))); x=self.pool(x)
        x=F.relu(self.b3(self.c3(x))); x=F.relu(self.b4(self.c4(x))); x=self.pool(x)
        x=self.drop(x); x=x.flatten(1); f=F.relu(self.fc1(x)); y=self.fc2(f)
        return (y,f) if return_features else y

class TinyCNN(nn.Module):
    def __init__(self,in_ch=1,n_classes=4):
        super().__init__(); self.c=nn.Conv2d(in_ch,16,3,padding=1); self.pool=nn.AdaptiveAvgPool2d((4,4)); self.fc1=nn.Linear(16*4*4,32); self.fc2=nn.Linear(32,n_classes)
    def forward(self,x,return_features=False):
        x=F.relu(self.c(x)); x=self.pool(x).flatten(1); f=F.relu(self.fc1(x)); y=self.fc2(f); return (y,f) if return_features else y


def build_model(dataset: str, input_shape, n_classes: int):
    if dataset in ('mnist','fashion_mnist','fashionmnist'): return MNISTCNN(n_classes)
    if dataset=='cifar10': return CIFARSmallCNN(n_classes)
    return TinyCNN(input_shape[0],n_classes)
