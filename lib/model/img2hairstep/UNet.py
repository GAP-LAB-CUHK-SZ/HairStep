import torch
import torch.nn as nn
from torch.nn import functional as F

class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        return torch.cat((x, r), 1)


class Body(nn.Module):
    def __init__(self,channels=16):
        super(Body, self).__init__()
        self.C1 = Conv(3, channels)
        self.D1 = DownSampling(channels)
        self.C2 = Conv(channels, channels*2)
        self.D2 = DownSampling(channels*2)
        self.C3 = Conv(channels*2, channels*4)
        self.D3 = DownSampling(channels*4)
        self.C4 = Conv(channels*4, channels*8)
        self.D4 = DownSampling(channels*8)
        self.C5 = Conv(channels*8, channels*16)

        self.U1 = UpSampling(channels*16)
        self.C6 = Conv(channels*16, channels*8)
        self.U2 = UpSampling(channels*8)
        self.C7 = Conv(channels*8, channels*4)
        self.U3 = UpSampling(channels*4)
        self.C8 = Conv(channels*4, channels*2)
        self.U4 = UpSampling(channels*2)
        self.C9 = Conv(channels*2, channels)

    def forward(self, I):
        R1 = self.C1(I)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        return O4

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.body=Body(16)
        self.pred = nn.Conv2d(16, 2, 3, 1, 1)

    def forward(self, I):
        O4=self.body(I)

        return self.pred(O4)

if __name__ == '__main__':
    from torchsummary import summary
    summary(Model(),(3,512,512),device='cpu')
