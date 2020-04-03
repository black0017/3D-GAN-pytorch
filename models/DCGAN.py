import torch
import torch.nn as nn
from torchsummary import summary

"""
DCGAN pytorch implementation based on https://arxiv.org/abs/1511.06434
"""


class Discriminator(torch.nn.Module):

    def __init__(self, in_channels=3, out_conv_channels=1024, dim=64):
        super(Discriminator, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim / 16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(out_conv_channels * self.out_dim * self.out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, self.out_conv_channels * self.out_dim * self.out_dim)
        x = self.out(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self, in_channels=1024, out_dim=64, out_channels=3, noise_dim=200):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 16)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.linear = torch.nn.Linear(noise_dim, in_channels * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels, out_channels=conv1_out_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=conv2_out_channels, out_channels=conv3_out_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(conv3_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.in_channels, self.in_dim, self.in_dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.out(x)


def test_dcgan():
    noise_dim = 100
    in_conv_channels = 512
    dim = 64  # cube volume
    model_generator = Generator(in_channels=in_conv_channels, out_dim=dim, out_channels=3, noise_dim=noise_dim)
    noise = torch.rand(1, noise_dim)
    generated_volume = model_generator(noise)
    print("Generator output shape", generated_volume.shape)
    model_discriminator = Discriminator(in_channels=3, dim=dim, out_conv_channels=in_conv_channels)
    out = model_discriminator(generated_volume)
    print("Discriminator output", out)
    print("Generator summary")
    summary(model_generator, (1, noise_dim))
    print("Discriminator summary")
    summary(model_discriminator, (3,64,64))

test_dcgan()