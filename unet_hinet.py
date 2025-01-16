from diffusers import UNet2DConditionModel
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.upsampling import Upsample2D
import torch
from torch import nn
from unhcv.common.utils import find_path

from unhcv.nn.models import Unet, ResNet, UnetConfig

from invblock import INV_block_v2
import hi_modules.module_util as mutil
from rrdb_denselayer import ResidualDenseBlock_out_v2


class ResnetBlock2D(ResnetBlock2D):
    def __init__(self, *args, **kwargs):
        in_channels = args[0]
        out_channels = args[1]
        super().__init__(in_channels=in_channels, out_channels=out_channels, temb_channels=None, **kwargs)
        mutil.initialize_weights([self.conv2], 0.)

    def forward(self, x, temb=None):
        return super().forward(x, temb)


class HiNetUnet(nn.Module):
    def __init__(self, channels_lt, num_inv_blocks_lt = (3, 3, 3, 3), output_channels=12, model_config={}):
        """

        Args:
            channels_lt: (32, 64, 128, 256)
        """
        super().__init__()
        self.shared_rev_blocks = model_config.get("shared_rev_blocks", True)
        subnet_constructor = globals()[model_config.get("subnet_constructor", "ResnetBlock2D")]
        num_inv_blocks_lt = model_config.get("num_inv_blocks_lt", num_inv_blocks_lt)
        self.unet = Unet(UnetConfig(channels_lt))
        self.inv_blocks = nn.ModuleList()
        for i in range(len(channels_lt)):
            inv_blocks = []
            channels = channels_lt[i]
            num_inv_blocks = num_inv_blocks_lt[i]
            for _ in range(num_inv_blocks):
                inv_blocks.append(INV_block_v2(subnet_constructor=subnet_constructor, in_1=channels, in_2=channels))
            self.inv_blocks.append(nn.ModuleList(inv_blocks))
        if not self.shared_rev_blocks:
            self.inv_blocks_1 = nn.ModuleList()
            for i in range(len(channels_lt)):
                inv_blocks = []
                channels = channels_lt[i]
                num_inv_blocks = num_inv_blocks_lt[i]
                for _ in range(num_inv_blocks):
                    inv_blocks.append(INV_block_v2(subnet_constructor=subnet_constructor, in_1=channels, in_2=channels))
                self.inv_blocks_1.append(nn.ModuleList(inv_blocks))
        self.backbone = ResNet("resnet50", in_channels=12, checkpoint=find_path("model/resnet/resnet50-0676ba61.pth"))
        self.upsample = nn.Sequential(Upsample2D(channels_lt[0], out_channels=channels_lt[0]//2, use_conv=True),
                                      Upsample2D(channels_lt[0]//2, out_channels=output_channels, use_conv=True))


    def c2b(self, x, rev=False):
        if not rev:
            x = torch.chunk(x, 2, 1)
            x = torch.cat(x, 0)
        else:
            x = torch.chunk(x, 2, 0)
            x = torch.cat(x, 1)
        return x

    def forward(self, x, rev=False):
        fpn_features = []
        x = self.c2b(x)
        for i in range(4):
            x = self.backbone.forward_stage(x)
            x = self.c2b(x, rev=True)
            if self.shared_rev_blocks:
                inv_blocks = self.inv_blocks[i]
            else:
                inv_blocks = self.inv_blocks_1[i] if rev else self.inv_blocks[i]
            for inv_block in inv_blocks:
                x = inv_block(x, rev=rev)
            x = self.c2b(x)
            fpn_features.append(x)

        unet_features = self.unet(fpn_features)
        out = self.upsample(unet_features[-1])
        out = self.c2b(out, rev=True)

        return out


if __name__ == '__main__':
    unet = HiNetUnet((256, 512, 1024, 1024), (1, ) * 4)
    print(unet)
    import torch
    x = torch.randn(1, 6, 512, 512)
    out = unet(x)
    breakpoint()
    print('Done')