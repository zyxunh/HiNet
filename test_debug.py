import PIL.Image
import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from kornia import tensor_to_image
from torchvision.transforms.functional import pil_to_tensor, to_tensor
from unhcv.common import write_im
from unhcv.common.image import visual_tensor
from unhcv.common.utils import find_path
from unhcv.common.utils.global_item import GLOBAL_ITEM

from model import *
import config as c
import datasets
import hi_modules.Unet_common as common

import numpy as np
from PIL import Image
import io


def compress_image_to_jpeg(image: PIL.Image.Image, quality=85):
    # Convert the image to RGB mode if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Save the image to a bytes buffer in JPEG format
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    return buffer


def decompress_image_from_jpeg(buffer):
    # Load the image from the bytes buffer
    image = Image.open(buffer)

    # Convert the image to a NumPy array
    # image_array = np.array(image)

    return image

def jpeg_degredation(image: PIL.Image.Image, quality=85):
    buffer = compress_image_to_jpeg(image, quality=quality)
    return decompress_image_from_jpeg(buffer)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

load(c.MODEL_PATH + c.suffix)

net.eval()

dwt = common.DWT()
iwt = common.IWT()


with torch.no_grad():
    for i, data in enumerate(datasets.testloader):
        data = data.to(device)
        cover = data[data.shape[0] // 2:, :, :, :]
        secret = data[:data.shape[0] // 2, :, :, :]
        secret = PIL.Image.open(find_path("code/CRoSS/asserts/1.png")).convert("RGB").resize((1024, 1024))
        secret = torchvision.transforms.ToTensor()(secret).unsqueeze(0).cuda()
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), 1)

        #################
        #    forward:   #
        #################
        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        outs = []
        for var in GLOBAL_ITEM.outs:
            outs.extend(var)
        breakpoint()
        outs = torch.cat(outs, dim=0)
        outs_iwt = iwt(outs)
        outs_iwt_show = visual_tensor(outs_iwt, max_value=1, min_value=0, row_column=(16, 2), reverse=True)
        write_im("/home/yixing/train_outputs/test.jpg", outs_iwt_show)

        steg_img = iwt(output_steg)
        steg_img_pil = steg_img[0].detach().cpu().clamp(min=0, max=1).permute(1, 2, 0).numpy()
        steg_img_pil = PIL.Image.fromarray((steg_img_pil * 255).round().astype(np.uint8))
        quality = 1000
        print(quality)
        buffer = compress_image_to_jpeg(steg_img_pil, quality=quality)
        steg_img = to_tensor(decompress_image_from_jpeg(buffer))[None].cuda()
        # steg_img = to_tensor(steg_img_pil)[None].cuda()
        output_steg = dwt(steg_img)

        backward_z = gauss_noise(output_z.shape)

        #################
        #   backward:   #
        #################
        output_rev = torch.cat((output_steg, backward_z), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
        cover_rev = iwt(cover_rev)
        resi_cover = (steg_img - cover).abs() * 20
        resi_secret = (secret_rev - secret).abs() * 20

        cats = torch.cat((cover, steg_img, resi_cover, secret, secret_rev, resi_secret), dim=3)
        torchvision.utils.save_image(cover, c.IMAGE_PATH_cover + '%.5d.jpg' % i)
        torchvision.utils.save_image(secret, c.IMAGE_PATH_secret + '%.5d.jpg' % i)
        torchvision.utils.save_image(steg_img, c.IMAGE_PATH_steg + '%.5d.jpg' % i)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_secret_rev + '%.5d.jpg' % i)
        torchvision.utils.save_image(cats, c.IMAGE_PATH_cats + '%.5d.jpg' % i)




