import torch
import math
import torch.nn.functional as F
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        #contracting path
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),)
        self.downconv1 = downStep(64, 128)
        self.downconv2 = downStep(128, 256)
        self.downconv3 = downStep(256, 512)
        self.downconv4 = downStep(512, 1024)  
        #expanding path
        self.upconv1 = upStep(1024, 512)
        self.upconv2 = upStep(512, 256)
        self.upconv3 = upStep(256, 128)
        self.upconv4 = upStep(128, 64, withReLU=False)
        self.opconv = nn.Conv2d(64, n_classes, 1) 
        
    def forward(self, x):
        # todo
        x_conv = self.conv(x)
        x_down1 = self.downconv1(x_conv)        
        x_down2 = self.downconv2(x_down1)
        x_down3 = self.downconv3(x_down2)
        x_down4 = self.downconv4(x_down3)
        #using the previous layers
        x_up1 = self.upconv1(x_down4,x_down3)
        x_up2 = self.upconv2(x_up1,x_down2)
        x_up3 = self.upconv3(x_up2,x_down1)
        x_up4 = self.upconv4(x_up3,x_conv)
        x = self.opconv(x_up4)
        
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.maxpool = nn.MaxPool2d(2)
        # double convoultion for each down step
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
        self.uppool = nn.ConvTranspose2d(inC, outC, 2, stride=2)
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
        x = self.uppool(x)
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
