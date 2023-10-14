import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            # input 1824
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv1d(512, 1024, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            #
            nn.Conv1d(1024, 2048, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, inplace=True),


            # state size 114
            nn.Conv1d(2048, 1, kernel_size=114, stride=1, padding=0, bias=False),
        )

    def forward(self, x, y=None):
        x = self.main(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
            
        self.main = nn.Sequential(
            nn.ConvTranspose1d(nz, 12*2048, 4, 2, 1, bias=False),
            nn.BatchNorm1d(12*2048),
            nn.ReLU(True),

            nn.ConvTranspose1d(12*2048, 12*1024,4, 2, 1, bias=False),
            nn.BatchNorm1d(12*1024),
            nn.ReLU(True),

            nn.ConvTranspose1d(12*1024, 12*512, 4, 2, 1,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            # nn.Conv1d(500, ),
            # nn.Tanh()
            nn.ConvTranspose1d(12*512, 12*256, 4, 2, 1,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(12*256, 12*128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(12*128, 12*64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.Conv1d(12*64, 12*1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        return x