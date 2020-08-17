import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(128, 256, 4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(True),

            nn.Conv2d(256, 1, 4, stride=1, padding=1, bias=False),
            nn.LeakyReLU(True)
        )

    def forward(self, x):
        x = self.conv(x)       # (batch, 128, 2, 2)
        return x