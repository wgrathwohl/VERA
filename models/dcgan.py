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
class EnergyModel_mnist(nn.Module):
    def __init__(self, input_dim=1, dim=512):
        super().__init__()
        self.expand = nn.Linear(2 * 2 * dim, 1)
        self.main = nn.Sequential(
            nn.Conv2d(input_dim, dim // 8, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, dim, 5, 2, 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        #self.apply(weights_init)

    def forward(self, x, return_fmap=False):
        out = self.main(x).view(x.size(0), -1)
        energies = self.expand(out).squeeze(-1)
        if return_fmap:
            return out, energies
        else:
            return energies
class EnergyModel(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, dim // 8, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 8, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 8, dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 4, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim // 2, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(dim // 2, dim, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.expand = nn.Linear(4 * 4 * dim, 1)
        #self.apply(weights_init)

    def forward(self, x):
        out = self.main(x).view(x.size(0), -1)
        return self.expand(out).squeeze(-1)

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

class Generator_mnist(nn.Module):
    def __init__(self, input_dim=1, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 2 * 2 * dim)
        self.main = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim // 2, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 5, 2, 2),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, input_dim, 5, 2, 2, output_padding=1),
            nn.Tanh(),
        )
        #self.apply(weights_init)

    def forward(self, z):
        x = self.expand(z).view(z.size(0), -1, 2, 2)
        return self.main(x)
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

class Generator(nn.Module):
    def __init__(self, z_dim=128, dim=512):
        super().__init__()
        self.expand = nn.Linear(z_dim, 4 * 4 * dim)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, 4, 2, 1),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 2, dim // 4, 4, 2, 1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 4, dim // 8, 4, 2, 1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim // 8, 3, 3, 1, 1),
            nn.Tanh(),
        )
        #self.apply(weights_init)

    def forward(self, z):
        out = self.expand(z).view(z.size(0), -1, 4, 4)
        return self.main(out)