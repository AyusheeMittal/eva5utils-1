import torch.nn as nn
import torch.nn.functional as F



class Model7(nn.Module):
    def __init__(self):
        super(Model7, self).__init__()
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''

        # self.convblock2 = nn.Sequential(
        # nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5,5), padding=1, dilation=1, bias=False),
        # nn.ReLU(),
        # nn.BatchNorm2d(128)
        # )
        self.depthwise1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3 ,3), padding=1, dilation=1, bias=False, groups=3)
        self.pointwise1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=25, kernel_size=(1 ,1)),
            nn.ReLU(),
            nn.BatchNorm2d(25)
        )

        self.depthwise2 = nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(3 ,3), padding=1, dilation=1, bias=False, groups=25)
        self.pointwise2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1 ,1)),
            nn.ReLU(),
            nn.BatchNorm2d(50)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        # self.convblock3 = nn.Sequential(
        # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5,5), padding=1, bias=False),
        # nn.ReLU(),
        # nn.BatchNorm2d(256)
        # )
        self.depthwise3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(3 ,3), padding=1, dilation=1, bias=False, groups=50)
        self.pointwise3 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1 ,1)),
            nn.ReLU(),
            nn.BatchNorm2d(100)
        )

        self.depthwise4 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(3 ,3), padding=1, dilation=1, bias=False, groups=100)
        self.pointwise4 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1 ,1)),
            nn.ReLU(),
            nn.BatchNorm2d(200)
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=500, kernel_size=(3 ,3), padding=1, dilation=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(500)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=500, out_channels=10, kernel_size=(1 ,1), padding=0, bias=False),
            nn.ReLU(),
            # nn.BatchNorm2d(20)
        )

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=3)
        ) # output_size = 1


    def forward(self, x):


        # x = self.convblock2(x)
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.pool2(x)
        # x = self.convblock3(x)
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.depthwise4(x)
        x = self.pointwise4(x)
        x = self.pool3(x)
        x = self.convblock1(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return x

