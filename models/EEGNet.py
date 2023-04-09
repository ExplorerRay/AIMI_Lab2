import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25))
        self.bn1 = nn.BatchNorm2d(16)

        # Layer 2
        self.depthwiseConv2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False, padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(32)
        self.activation1 = nn.ELU()
        self.avgPool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # Layer 3
        self.separableConv3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), bias=False, padding=(0, 7))
        self.bn3 = nn.BatchNorm2d(32)
        self.activation2 = nn.ELU()
        self.avgPool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))

        # FC Layer
        self.fc = nn.Linear(736, 2)

    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.avgPool1(x)

        # Layer 2
        x = self.depthwiseConv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.avgPool2(x)

        # Layer 3
        x = self.separableConv3(x)
        x = self.bn3(x)
        x = self.activation2(x)
        x = self.avgPool2(x)

        # FC Layer
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
        
