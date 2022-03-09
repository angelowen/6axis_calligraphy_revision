import torch
import torch.nn as nn
from torchvision.models import vgg19


class FeatureExtractor(nn.Module):
    r"""
    build for content loss
    using vgg19 0~18 layer as feature extractor

    input: tensor
    output: tensor
    """
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(
            *list(model.features.children())[:18]
        )
        self.feature_extractor[0] = nn.Conv2d(1, 64, 3, padding=1)

    def forward(self, x):
        return self.feature_extractor(x)


class ResBlock(nn.Module):
    """
    Build the Residual Block of SRGAN
    """
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(in_channel, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(in_channel, 0.8)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    Build the Generator of SRGAN

    inputs:
        in_channel: input channel
        out_channel: output channel
        n_resblock: number of residual blocks

    output:
    """
    def __init__(self, in_channel=1, out_channel=1, n_resblock=16):
        super(Generator, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=9, padding=9//2),
            nn.PReLU()
        )

        # Residual Blocks
        blocks = [ResBlock(64)] * n_resblock

        # blocks = []
        # for _ in range(n_resblock):
            # blocks.append(ResBlock(64))

        self.res_blocks = nn.Sequential(*blocks)

        # Second conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=3//2),
            nn.BatchNorm2d(64, 0.8)
        )

        # upsampling layers
        upsampling = []
        for _ in range(2):
            upsampling.append(nn.Conv2d(64, 256, 3, 1, 1))
            upsampling.append(nn.BatchNorm2d(256))
            upsampling.append(nn.PixelShuffle(upscale_factor=2))
            upsampling.append(nn.PReLU())

        self.upsampling = nn.Sequential(*upsampling)

        # output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_channel, kernel_size=9, padding=9//2),
            nn.Tanh()
        )

    def forward(self, x):
        out_1 = self.conv1(x)
        out = self.res_blocks(out_1)

        out_2 = self.conv2(out)
        out = torch.add(out_1, out_2)

        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()

        layers = []
        in_features = in_channel
        blocks = [64, 128, 256, 512]
        for i, out_features in enumerate(blocks):
            layers.extend(
                self.discriminator_block(in_features, out_features, first=(i == 0))
            )
            in_features = out_features

        """
        self.output_layer = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        """
        # using conv2d instead of linear
        layers.append(
            nn.Conv2d(out_features, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

        self.dis_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.dis_block(x)


    @staticmethod
    def discriminator_block(in_features, out_feature, first=False):
        layers = []

        # first block
        layers.append(nn.Conv2d(in_features, out_feature, 3, stride=1, padding=1))
        # first layer don't need BN
        if not first:
            layers.append(nn.BatchNorm2d(out_feature, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # second block
        layers.append(nn.Conv2d(out_feature, out_feature, 3, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(out_feature, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        return layers


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, y, yhat):
        return torch.sqrt(self.mse(y, yhat) + self.eps)


if __name__ == '__main__':
    print(Discriminator(1))