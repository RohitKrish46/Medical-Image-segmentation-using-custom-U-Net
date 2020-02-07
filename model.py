import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),)
        self.conv2 = downStep(64, 128)
        self.conv3 = downStep(128, 256)
        self.conv4 = downStep(256, 512)
        self.conv5 = downStep(512, 1024)  
        self.conv6 = upStep(1024, 512)
        self.conv7 = upStep(512, 256)
        self.conv8 = upStep(256, 128)
        self.conv9 = upStep(128, 64, withReLU=False)
        self.conv10 = nn.Conv2d(64, n_classes, 1)  # last convolutional layer 
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        # todo
        x1 = self.conv1(x)
        x2 = self.conv2(x1)        
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5,x4)
        x7 = self.conv7(x6,x3)
        x8 = self.conv8(x7,x2)
        x9 = self.conv9(x8,x1)
        x10 = self.conv10(x9)
        x = self.sigmoid1(x10)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.maxpool = nn.MaxPool2d(2)
        self.downconv = nn.Sequential(
            nn.Conv2d(inC, outC, 3), nn.BatchNorm2d(outC), nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3), nn.BatchNorm2d(outC), nn.ReLU(inplace=True),)
    def forward(self, x):
        # todo  
        x = self.maxpool(x)
        x = self.downconv(x)   
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!        
        self.uppooling = nn.ConvTranspose2d(inC, outC, 2, stride=2)
        if(withReLU==True):
            self.upConv = nn.Sequential(
            nn.Conv2d(inC, outC, 3), nn.BatchNorm2d(outC), nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, 3), nn.BatchNorm2d(outC), nn.ReLU(inplace=True),)
        else:
            self.upConv = nn.Sequential(
                nn.Conv2d(inC, outC, 3), nn.BatchNorm2d(outC), 
                nn.Conv2d(outC, outC, 3), nn.BatchNorm2d(outC),)
    def forward(self, x, x_down):
        # todo
        x = self.uppooling(x)
        #getting the dimensions to pad
        Ypad = x.size()[2] - x_down.size()[2]
        Xpad = x.size()[3] - x_down.size()[3]
        pad_left = math.floor(Xpad / 2)
        pad_right = Xpad-math.floor(Xpad / 2)
        pad_top = math.floor(Ypad / 2)
        pad_bot = Ypad-math.floor(Ypad / 2)
        x_down = F.pad(x_down, pad=(pad_left, pad_right, pad_top, pad_bot))
        x = torch.cat([x_down, x], dim=1)
        x = self.upConv(x)
        return x
