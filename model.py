import torch.optim
import torch.nn as nn
import config as c
from hinet import Hinet


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.inv_on = getattr(config, 'inv_on', True)
        if self.inv_on:
            self.model = Hinet()
        else:
            self.model1 = Hinet()
            self.model2 = Hinet()

    def forward(self, x, rev=False):

        if self.inv_on:
            if not rev:
                out = self.model(x)

            else:
                out = self.model(x, rev=True)
        else:
            if not rev:
                out = self.model1(x)
            else:
                out = self.model2(x, rev=True)

        return out


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
