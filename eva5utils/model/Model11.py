import torch.nn as nn
import torch.nn.functional as F
import torch


class Model11(nn.Module):
    def __init__(self):
        super(Model11, self).__init__()

        # PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 1
        # X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        self.x1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        self.r1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Layer 2
        self.l2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Layer 3
        # X2 = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        self.x2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        self.r2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.mp = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10)

    def forward(self, inp):
        x0 = self.prep(inp)
        x1 = self.x1(x0)
        r1 = self.r1(x1)
        l1 = torch.add(x1, r1)

        l2 = self.l2(l1)

        x2 = self.x2(l2)
        r2 = self.r2(x2)
        l3 = torch.add(x2, r2)

        mp = self.mp(l3)
        flatten = mp.squeeze()
        fc = self.fc(flatten)
        # print("fc: ", fc.shape)

        y = F.softmax(fc, dim=1)
        return y