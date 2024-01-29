import os.path
import sys
import os
import warnings
import torch

if torch.cuda.is_available():
    os.environ['COMMANDLINE_ARGS'] = "--upcast-sampling --skip-torch-cuda-test --no-half-vae interrogate"
elif torch.backends.mps.is_available():
    os.environ['COMMANDLINE_ARGS'] = "--skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate"
else:
    os.environ['COMMANDLINE_ARGS'] = "--skip-torch-cuda-test --no-half-vae --no-half interrogate"
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
os.environ['TORCH_COMMAND'] = "pip install torch==2.0.1 torchvision==0.15.2"
os.environ['ERROR_REPORTING'] = "FALSE"
os.environ['PIP_IGNORE_INSTALLED'] = "0"
os.environ['SD_WEBUI_RESTART'] = "tmp/restart"

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")

warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

from ..modules import sd_samplers
sd_samplers.set_samplers()
from ..modules import shared_init
shared_init.initialize()

from ..modules import sd_models
sd_models.setup_model()

from ..modules import devices
devices.first_time_calculation()

from  ..modules import shared as shared
warnings.filterwarnings("default" if shared.opts.show_warnings else "ignore", category=UserWarning)


import io
from PIL import Image, PngImagePlugin
import base64
import piexif
from ..modules import processing, images, devices
import math

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from ..modules.shared import cmd_opts


real_esrgan_info = {
    "R-ESRGAN General 4xV3": {
        "scale": 4, 
        "model":lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    },
    "R-ESRGAN General WDN 4xV3": {
        "scale": 4, 
        "model":lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu'), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth"
    },
    "R-ESRGAN AnimeVideo": {
        "scale": 4, 
        "model":lambda: SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    },
    "R-ESRGAN 4x+ Anime6B": {
        "scale": 4, 
        "model": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    }, 
    "R-ESRGAN 4x+": {
        "scale": 4, 
        "model": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    }, 
    "R-ESRGAN 2x+": {
        "scale": 4, 
        "model": lambda: RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 
        "path":"https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    }, 
}

def download_realesrgan(model_id, weight_paths):
    import requests
    if model_id not in real_esrgan_info: 
        raise Exception("Invalid model id")
    download_url = real_esrgan_info[model_id]['path']
    print("download url", download_url)
    response = requests.get(download_url)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    with open(weight_paths, 'wb') as f:
        f.write(response.content)

class RealEsrganPipeline:
    def __init__(self, model_id, model_path=None):
        self.model_weights = model_path
        model_dict = real_esrgan_info[model_id]
        self.scale = model_dict['scale']
        # if model_path is None:

        self.__upsampler = RealESRGANer(
            scale=model_dict['scale'], 
            model_path=model_path, 
            model=model_dict['model'](), 
            half=not cmd_opts.no_half and not cmd_opts.upcast_sampling,
            tile=shared.opts.ESRGAN_tile,
            tile_pad=shared.opts.ESRGAN_tile_overlap,
            device=shared.device
        )

    def upscale(self, img: Image, scale) -> Image:
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(3):
            if img.width >= dest_w and img.height >= dest_h:
                break

            shape = (img.width, img.height)

            img = self.__upsampler.enhance(np.array(img), outscale=self.scale)[0]
            img = Image.fromarray(img)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img


import sys

import numpy as np
import torch
from PIL import Image

from ..modules import esrgan_model_arch as arch
from ..modules import images, devices
from tqdm import tqdm

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
NEAREST = (Image.Resampling.NEAREST if hasattr(Image, 'Resampling') else Image.NEAREST)

class EsrganPipeline:
    def __init__(self, model_path):
        self.model_weights = model_path
        self.__model = self.__load_model(self.model_weights)
        self.__model = self.__model.to(devices.device_esrgan)

    def __load_model(self, path: str):
        filename = path

        state_dict = torch.load(filename, map_location='cpu' if devices.device_esrgan.type == 'mps' else None)

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            num_conv = 16 if "realesr-animevideov3" in filename else 32
            model = arch.SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=num_conv, upscale=4, act_type='prelu')
            model.load_state_dict(state_dict)
            model.eval()
            return model

        if "body.0.rdb1.conv1.weight" in state_dict and "conv_first.weight" in state_dict:
            nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
            state_dict = _resrgan2normal(state_dict, nb)
        elif "conv_first.weight" in state_dict:
            state_dict = _mod2normal(state_dict)
        elif "model.0.weight" not in state_dict:
            raise Exception("The file is not a recognized ESRGAN model.")

        in_nc, out_nc, nf, nb, plus, mscale = _infer_params(state_dict)

        model = arch.RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus)
        model.load_state_dict(state_dict)
        model.eval()

        return model

    def upscale(self, img: Image, scale: int = 2) -> Image:
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(3):
            if img.width >= dest_w and img.height >= dest_h:
                break

            shape = (img.width, img.height)

            img = _esrgan_upscale(model=self.__model, img=img)

            if shape == (img.width, img.height):
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img
    
def _upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(devices.device_esrgan)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255. * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, 'RGB')

def _esrgan_upscale(model, img):
    if shared.opts.ESRGAN_tile == 0:
        return _upscale_without_tiling(model, img)

    grid = images.split_grid(img, shared.opts.ESRGAN_tile, shared.opts.ESRGAN_tile, shared.opts.ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1

    # for y, h, row in grid.tiles:
    for y, h, row in tqdm(grid.tiles, desc="Processing Tiles"):
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = _upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = images.Grid(newtiles, grid.tile_w * scale_factor, grid.tile_h * scale_factor, grid.image_w * scale_factor, grid.image_h * scale_factor, grid.overlap * scale_factor)
    output = images.combine_grid(newgrid)
    return output

def _mod2normal(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    if 'conv_first.weight' in state_dict:
        crt_net = {}
        items = list(state_dict)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if 'RDB' in k:
                ori_k = k.replace('RRDB_trunk.', 'model.1.sub.')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net['model.1.sub.23.weight'] = state_dict['trunk_conv.weight']
        crt_net['model.1.sub.23.bias'] = state_dict['trunk_conv.bias']
        crt_net['model.3.weight'] = state_dict['upconv1.weight']
        crt_net['model.3.bias'] = state_dict['upconv1.bias']
        crt_net['model.6.weight'] = state_dict['upconv2.weight']
        crt_net['model.6.bias'] = state_dict['upconv2.bias']
        crt_net['model.8.weight'] = state_dict['HRconv.weight']
        crt_net['model.8.bias'] = state_dict['HRconv.bias']
        crt_net['model.10.weight'] = state_dict['conv_last.weight']
        crt_net['model.10.bias'] = state_dict['conv_last.bias']
        state_dict = crt_net
    return state_dict

def _resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        re8x = 0
        crt_net = {}
        items = list(state_dict)

        crt_net['model.0.weight'] = state_dict['conv_first.weight']
        crt_net['model.0.bias'] = state_dict['conv_first.bias']

        for k in items.copy():
            if "rdb" in k:
                ori_k = k.replace('body.', 'model.1.sub.')
                ori_k = ori_k.replace('.rdb', '.RDB')
                if '.weight' in k:
                    ori_k = ori_k.replace('.weight', '.0.weight')
                elif '.bias' in k:
                    ori_k = ori_k.replace('.bias', '.0.bias')
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net[f'model.1.sub.{nb}.weight'] = state_dict['conv_body.weight']
        crt_net[f'model.1.sub.{nb}.bias'] = state_dict['conv_body.bias']
        crt_net['model.3.weight'] = state_dict['conv_up1.weight']
        crt_net['model.3.bias'] = state_dict['conv_up1.bias']
        crt_net['model.6.weight'] = state_dict['conv_up2.weight']
        crt_net['model.6.bias'] = state_dict['conv_up2.bias']

        if 'conv_up3.weight' in state_dict:
            # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
            re8x = 3
            crt_net['model.9.weight'] = state_dict['conv_up3.weight']
            crt_net['model.9.bias'] = state_dict['conv_up3.bias']

        crt_net[f'model.{8+re8x}.weight'] = state_dict['conv_hr.weight']
        crt_net[f'model.{8+re8x}.bias'] = state_dict['conv_hr.bias']
        crt_net[f'model.{10+re8x}.weight'] = state_dict['conv_last.weight']
        crt_net[f'model.{10+re8x}.bias'] = state_dict['conv_last.bias']

        state_dict = crt_net
    return state_dict

def _infer_params(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if (part_num > scalemin
                and parts[0] == "model"
                and parts[2] == "weight"):
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2 ** scale2x

    return in_nc, out_nc, nf, nb, plus, scale
