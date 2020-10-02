"""
Architectures from DCGAN.
"""

import torch.nn as nn


def DCGANDiscriminator(in_channels=3, ngf=64, nout=1, img_size=32, weight_norm=False):
    """
    DCGAN Discriminator.
    """
    if img_size == 32:
        final_kernel = 2
    elif img_size == 64:
        final_kernel = 4
    else:
        raise ValueError
    if weight_norm:
        return nn.Sequential(
            # input is (nc) x 32 x 32
            nn.utils.weight_norm(nn.Conv2d(in_channels, ngf, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.utils.weight_norm(nn.Conv2d(ngf, 2 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 8 x 8
            nn.utils.weight_norm(nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 4 x 4
            nn.utils.weight_norm(nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1)),
            # nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 2 x 2
            nn.utils.weight_norm(nn.Conv2d(8 * ngf, nout, final_kernel, 1, 0, bias=False)),
            nn.Flatten(start_dim=1)
        )
    else:
        return nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(in_channels, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 16 x 16
            nn.Conv2d(ngf, 2 * ngf, 4, 2, 1),
            # nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 8 x 8
            nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1),
            # nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 4 x 4
            nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1),
            # nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 2 x 2
            nn.Conv2d(8 * ngf, nout, final_kernel, 1, 0, bias=False),
            nn.Flatten(start_dim=1)
        )


def BNDCGANDiscriminator(in_channels=3, ngf=64, nout=1):
    """
    DCGAN Discriminator with batchnorm.
    """
    return nn.Sequential(
        # input is (nc) x 32 x 32
        nn.Conv2d(in_channels, ngf, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf) x 16 x 16
        nn.Conv2d(ngf, 2 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(2 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*2) x 8 x 8
        nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(4 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*4) x 4 x 4
        nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(8 * ngf),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (ngf*8) x 2 x 2
        nn.Conv2d(8 * ngf, nout, 2, 1, 0),
        nn.Flatten(start_dim=1)
    )


def DCGANGenerator(noise_dim, unit_interval, out_channels=3, ngf=64, img_size=32):
    """
    DCGan Generator.
    """
    class G(nn.Module):
        """
        Generator torch module.
        """
        def __init__(self):
            super().__init__()
            if unit_interval:
                final_act = nn.Sigmoid()
            else:
                final_act = nn.Tanh()
            if img_size == 32:
                first_kernel = 2
            elif img_size == 64:
                first_kernel = 4
            else:
                raise ValueError
            self.first = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(noise_dim, ngf * 8, first_kernel, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1),
                final_act
            )

        def forward(self, x):
            """
            Forward pass.
            """
            x = x.view(x.size(0), -1, 1, 1)
            x = self.first(x)
            return x
    return G()
