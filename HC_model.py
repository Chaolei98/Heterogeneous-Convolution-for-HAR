import torch
import torch.nn as nn
import torch.nn.functional as F


class HConv(nn.Module):
    def __init__(self, input_channel, kernel_size=(5,1), stride=1, padding=(1,0), pooling_r=(5,1)):
        super(HConv, self).__init__()

        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding),
                    nn.BatchNorm2d(input_channel),
                    nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(F.interpolate(self.k2(x), identity.size()[2:]))
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

#----------CNN----------
class HCBlock(nn.Module):
    def __init__(self, input_channel, kernel_size, stride, padding):
        super(HCBlock, self).__init__()

        self.HC = HConv(input_channel//2, kernel_size, stride, padding)
        self.k1 = nn.Sequential(
            nn.Conv2d(input_channel//2, input_channel//2, kernel_size, stride, padding),
            nn.BatchNorm2d(input_channel//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out_a, out_b = torch.split(x, [x.size(1)//2, x.size(1)//2], dim=1)
        out_a = self.k1(out_a)
        out_b = self.HC(out_b)
        out = torch.cat([out_a, out_b], dim=1)

        return out


class CNN_HC(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(CNN_HC, self).__init__()

        self.layer1 = self._make_layers(input_channel, 64, (6, 1), (3, 1), (1, 0))
        self.layer2 = self._make_layers(64, 64, (3, 1), (1, 1), (1, 0),HC=True)
        self.layer3 = self._make_layers(64, 128, (6, 1), (3, 1), (1, 0))
        self.fc = nn.Linear(128 * 18 * 40, num_classes)

    def _make_layers(self, input_channel, output_channel, kernel_size, stride, padding, HC=False):
        if HC == True:
            return HCBlock(input_channel, kernel_size, stride, padding)
        else:
            return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)

        return out


#----------ResNet----------
class HCBottleneck(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super(HCBottleneck, self).__init__()

        self.HC = HConv(output_channel//2, (3,1), 1, (1,0))
        self.k1 = nn.Sequential(
            nn.Conv2d(output_channel//2, output_channel//2, (3,1), 1, (1,0)),
            nn.BatchNorm2d(output_channel//2),
            nn.ReLU(inplace=True)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        out_a, out_b = torch.split(x, [x.size(1)//2, x.size(1)//2], dim=1)
        out_a = self.k1(out_a)
        out_b = self.HC(out_b)
        out = torch.cat([out_a, out_b], dim=1)
        return out

class Resnet_HC(nn.Module):
    def __init__(self, input_channel, num_classes):
        super(Resnet_HC, self).__init__()
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU(True))
        self.HC1 = HCBottleneck(64, 64, (3,1), (1,1), (1, 0))
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(64))


        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(True))
        self.HC2 = HCBottleneck(128, 128, (3,1), (1,1), (1, 0))
        self.shortcut2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128))


        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(True))
        self.shortcut3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(6, 1), stride=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256))
        self.fc = nn.Sequential(
            nn.Linear(51200, num_classes))

    def forward(self, x):
        out1 = self.Block1(x)
        out1 = self.HC1(out1)
        y1 = self.shortcut1(x)
        out = y1 + out1

        out2 = self.Block2(out)
        out2 = self.HC2(out2)
        y2 = self.shortcut2(out)
        out = y2 + out2

        out3 = self.Block3(out)
        y3 = self.shortcut3(out)
        out = y3 + out3

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = nn.LayerNorm(out.size())(out.cpu())
        out = out.cuda()
        return out