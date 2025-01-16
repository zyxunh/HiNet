import torch
import torch.nn as nn
import hi_modules.module_util as mutil


# Dense connection
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5


class ResidualDenseBlock_out_v2(ResidualDenseBlock_out):
    def __init__(self, input, output, bias=True, kernel_size=1):
        super(ResidualDenseBlock_out, self).__init__()
        padding = kernel_size // 2
        mid_channels = max(32, output)
        self.conv1 = nn.Conv2d(input, mid_channels, kernel_size, 1, padding, bias=bias)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, padding, bias=bias)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, padding, bias=bias)
        self.conv4 = nn.Conv2d(mid_channels, mid_channels, kernel_size, 1, padding, bias=bias)
        self.conv5 = nn.Conv2d(mid_channels, output, kernel_size, 1, padding, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def res(self, x, i):
        return x + self.lrelu(getattr(self, f"conv{i}")(x))

    def forward(self, x):
        for i in range(1, 5):
            x = self.res(x, i)
        x = self.conv5(x)
        return x