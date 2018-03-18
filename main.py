import os.path as osp

import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from lith.trainers.classification import Trainer

root = osp.expanduser("~/torch_data")
normMean = [0.49139968, 0.48215827, 0.44653124]
normStd = [0.24703233, 0.24348505, 0.26158768]
normTransform = transforms.Normalize(normMean, normStd)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normTransform
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

train_data = datasets.CIFAR10(root, train=True, download=True, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=32, num_workers=1)

valid_data = datasets.CIFAR10(root, train=False, download=True, transform=valid_transform)
valid_loader = DataLoader(train_data, batch_size=32, num_workers=1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        self.linear = nn.Linear(24, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = Model()
optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)

trainer = Trainer(model, train_loader, valid_loader, optim, resume=True)
trainer.run()