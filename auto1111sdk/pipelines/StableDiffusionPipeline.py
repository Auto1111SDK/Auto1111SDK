import os.path
import sys
import os
import warnings
import torch
from .EsrganPipelines import EsrganPipeline, RealEsrganPipeline

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

from ..modules import script_callbacks, sd_hijack_optimizations, sd_hijack
script_callbacks.on_list_optimizers(sd_hijack_optimizations.list_optimizers)
sd_hijack.list_optimizers()

from ..modules import shared
warnings.filterwarnings("default" if shared.opts.show_warnings else "ignore", category=UserWarning)

from ..modules.processing import StableDiffusionProcessingTxt2Img, process_images, StableDiffusionProcessingImg2Img 
import io
from PIL import Image, PngImagePlugin, ImageDraw
import base64
import piexif
from ..modules import processing, images, devices, script_callbacks
import math
import random

default_args_txt2img = {
    'prompt': '',
    'negative_prompt': '',
    'styles': None,
    'seed': -1,
    'subseed': -1,
    'subseed_strength': 0,
    'seed_resize_from_h': -1,
    'seed_resize_from_w': -1,
    'sampler_name': 'Euler',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 20,
    'cfg_scale': 7.0,
    'width': 512,
    'height': 512,
    'restore_faces': None,
    'tiling': None,
    'do_not_save_samples': True,
    'do_not_save_grid': True,
    'eta': None,
    'denoising_strength': None,
    's_min_uncond': None,
    's_churn': None,
    's_tmax': None,
    's_tmin': None,
    's_noise': None,
    'override_settings': None,
    'override_settings_restore_afterwards': True,
    'refiner_checkpoint': None,
    'refiner_switch_at': None,
    'disable_extra_networks': False,
    'comments': None,
    'enable_hr': False,
    'firstphase_width': 0,
    'firstphase_height': 0,
    'hr_scale': 2.0,
    'hr_upscaler': None,
    'hr_second_pass_steps': 0,
    'hr_resize_x': 0,
    'hr_resize_y': 0,
    'hr_checkpoint_name': None,
    'hr_sampler_name': None,
    'hr_prompt': '',
    'hr_negative_prompt': '',
    'sampler_index': None, 
    'custom_upscaler': None
}

default_args_img2img = {
    'prompt': '',
    'init_images': [],
    'negative_prompt': '',
    'styles': None,
    'seed': -1,
    'subseed': -1,
    'subseed_strength': 0,
    'seed_resize_from_h': -1,
    'seed_resize_from_w': -1,
    'sampler_name': 'Euler',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 5,
    'cfg_scale': 7.0,
    'width': 512,
    'height': 768,
    'restore_faces': None,
    'tiling': None,
    'do_not_save_samples': True,
    'do_not_save_grid': True,
    'eta': None,
    'denoising_strength': 0.75,
    's_min_uncond': None,
    's_churn': None,
    's_tmax': None,
    's_tmin': None,
    's_noise': None,
    'override_settings': None,
    'override_settings_restore_afterwards': True,
    'refiner_checkpoint': None,
    'refiner_switch_at': None,
    'disable_extra_networks': False,
    'comments': None,
    'resize_mode': 1,
    'image_cfg_scale': None,
    'mask': None,
    'mask_blur_x': 4,
    'mask_blur_y': 4,
    'mask_blur': None,
    'inpainting_fill': 0,
    'inpaint_full_res': True,
    'inpaint_full_res_padding': 0,
    'inpainting_mask_invert': 0,
    'initial_noise_multiplier': None,
    'latent_mask': None,
    'sampler_index': None
}

default_args_img2img_inpainting = {
    'prompt': '',
    'init_images': [],
    'negative_prompt': '',
    'styles': None,
    'seed': -1,
    'subseed': -1,
    'subseed_strength': 0,
    'seed_resize_from_h': -1,
    'seed_resize_from_w': -1,
    'sampler_name': 'Euler',
    'batch_size': 1,
    'n_iter': 1,
    'steps': 5,
    'cfg_scale': 7.0,
    'restore_faces': None,
    'tiling': None,
    'do_not_save_samples': True,
    'do_not_save_grid': True,
    'eta': None,
    'denoising_strength': 0.75,
    's_min_uncond': None,
    's_churn': None,
    's_tmax': None,
    's_tmin': None,
    's_noise': None,
    'override_settings': None,
    'override_settings_restore_afterwards': True,
    'refiner_checkpoint': None,
    'refiner_switch_at': None,
    'disable_extra_networks': False,
    'comments': None,
    'resize_mode': 1,
    'image_cfg_scale': None,
    'mask': None,
    'mask_blur_x': 4,
    'mask_blur_y': 4,
    'mask_blur': None,
    'inpainting_fill': 1,
    'inpaint_full_res': True,
    'inpaint_full_res_padding': 0,
    'inpainting_mask_invert': 0,
    'initial_noise_multiplier': None,
    'latent_mask': None,
    'sampler_index': None
}

class StableDiffusionPipeline:
    def __init__(self, model_path, clip_skip = 1):
        self.__aliases = sd_models.list_models(model_path)
        self.weights_file = os.path.basename(model_path)
        self.__model_data = sd_models.SdModelData(self.__aliases)
        self.__pipe = sd_models.load_model(aliases=self.__aliases, model_data=self.__model_data, weights_file=self.weights_file)

    def __encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())        
        return [img_str.decode('utf-8')]
    
    def __process_args_txt2img(self, **kwargs):
        if "prompt" not in kwargs:
            raise TypeError("Missing Prompt")
        if 'num_images' in kwargs:
            kwargs['n_iter'] = kwargs.pop('num_images')
        invalid_keys = [key for key in kwargs if key not in default_args_txt2img]
        if invalid_keys:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_keys)}")
        final_args = {**default_args_txt2img, **kwargs}
        return final_args
    
    def __process_args_inpainting(self, **kwargs):
        if "prompt" not in kwargs:
            raise TypeError("Missing Prompt")
        if "init_image" not in kwargs:
            raise TypeError("Missing input image")
        if 'num_images' in kwargs:
            kwargs['n_iter'] = kwargs.pop('num_images')
        kwargs['init_images'] = self.__encode_image(kwargs['init_image'])
        del kwargs['init_image']
        invalid_keys = [key for key in kwargs if key not in default_args_img2img_inpainting and (key != "init_image")]
        if invalid_keys:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_keys)}")
        final_args = {**default_args_img2img_inpainting, **kwargs}
        return final_args

    def generate_txt2img(self, prompt: str, negative_prompt: str = "", seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, width: int = 512, height: int = 512, num_images: int = 1, sampler_name: str = 'Euler'):
        input_params = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'width': width,
            'height': height,
            'num_images': num_images,
            'sampler_name': sampler_name
        }

        input_params = self.__process_args_txt2img(**input_params)
        p = StableDiffusionProcessingTxt2Img(sd_model=self.__pipe, **input_params)
        p.is_api = True
        processed = process_images(p, self.__aliases, self.__model_data, self.__pipe, self.weights_file)
        if hasattr(p, 'close'):
            p.close()

        output_images = list(processed.images)
        return output_images
    
    def __process_args_img2img(self, **kwargs):
        if "prompt" not in kwargs:
            raise TypeError("Missing Prompt")
        if "init_image" not in kwargs:
            raise TypeError("Missing input image")
        if 'num_images' in kwargs:
            kwargs['n_iter'] = kwargs.pop('num_images')

        kwargs['init_images'] = self.__encode_image(kwargs['init_image'])
        del kwargs['init_image']

        invalid_keys = [key for key in kwargs if key not in default_args_img2img and (key != "init_image")]
        if invalid_keys:
            raise ValueError(f"Invalid parameter(s): {', '.join(invalid_keys)}")
        final_args = {**default_args_img2img, **kwargs}
        return final_args
    
    def generate_img2img(self, prompt: str, init_image: Image, negative_prompt: str = "", height: int = 512, width: int = 512, seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, num_images: int = 1, sampler_name: str = 'Euler', denoising_strength: float = 0.75):
        input_params = {
            'prompt': prompt,
            'init_image': init_image,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'width': width,
            'height': height,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'num_images': num_images,
            'sampler_name': sampler_name,
            'denoising_strength': denoising_strength
        }
        
        input_params = self.__process_args_img2img(**input_params)
        p = StableDiffusionProcessingImg2Img(sd_model=self.__pipe, **input_params)
        p.init_images = [init_image]
        p.is_api = True
        processed = process_images(p, self.__aliases, self.__model_data, self.__pipe, self.weights_file)
        if hasattr(p, 'close'):
            p.close()
        output_images = list(processed.images)
        return output_images
    
    def inpainting_img2img(self, mask: Image, init_image: Image, prompt: str, negative_prompt: str = "", seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, num_images: int = 1, sampler_name: str = 'Euler', denoising_strength: float = 0.75, mask_blur: int = 4, inpaint_full_res_padding: int =32):
        input_params = {
            'mask': mask, 
            'init_image': init_image, 
            'prompt': prompt, 
            'negative_prompt': negative_prompt, 
            'seed': seed, 
            'steps': steps, 
            'cfg_scale': cfg_scale, 
            'num_images': num_images, 
            'sampler_name': sampler_name, 
            'denoising_strength': denoising_strength, 
            'mask_blur': mask_blur, 
            'inpaint_full_res_padding': inpaint_full_res_padding
        }
        input_params = self.__process_args_inpainting(**input_params)
        p = StableDiffusionProcessingImg2Img(sd_model=self.__pipe, **input_params)
        p.is_api = True
        p.init_images = [init_image]
        p.image_mask = mask
        processed = process_images(p, self.__aliases, self.__model_data, self.__pipe, self.weights_file)
        if hasattr(p, 'close'):
            p.close()
        output_images = list(processed.images)
        return output_images
    
    def sd_upscale_img2img(self, prompt: str, init_image: Image, upscaler = None, overlap: int = 64, scale_factor: int = 2, negative_prompt: str = "", seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, num_images: int = 1, sampler_name: str = 'Euler', denoising_strength: float = 0.75):
        if isinstance(upscaler, (EsrganPipeline, RealEsrganPipeline)):
            raise ValueError("Upscaler should either be an instance of RealEsrgan pipeline or Esrgan pipeline")
        input_params = { 
            'init_images': [init_image], 
            'prompt': prompt, 
            'negative_prompt': negative_prompt, 
            'seed': seed, 
            'steps': steps, 
            'cfg_scale': cfg_scale, 
            'num_images': num_images, 
            'sampler_name': sampler_name, 
            'denoising_strength': denoising_strength, 
        }
        input_params = self.__process_args_img2img(**input_params)
        p = StableDiffusionProcessingImg2Img(sd_model=self.__pipe, **input_params)
        p.is_api = True
        p.init_images = [init_image]
        processing.fix_seed(p)

        p.extra_generation_params["SD upscale overlap"] = overlap
        p.extra_generation_params["SD upscale upscaler"] = "None"

        initial_info = None
        seed = p.seed

        init_img = p.init_images[0]
        init_img = images.flatten(init_img, shared.opts.img2img_background_color)
        if upscaler == None:
            img = init_img
        else:
            img = upscaler.upscale(img=init_img, scale=scale_factor)

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=overlap)

        batch_size = p.batch_size
        upscale_count = p.n_iter
        p.n_iter = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []

        for _y, _h, row in grid.tiles:
            for tiledata in row:
                work.append(tiledata[2])

        batch_count = math.ceil(len(work) / batch_size)

        result_images = []
        for n in range(upscale_count):
            start_seed = seed + n
            p.seed = start_seed

            work_results = []
            for i in range(batch_count):
                p.batch_size = batch_size
                p.init_images = work[i * batch_size:(i + 1) * batch_size]

                processed = processing.process_images(p, self.__aliases, self.__model_data, self.__pipe, self.weights_file) 

                if initial_info is None:
                    initial_info = processed.info

                p.seed = processed.seed + 1
                work_results += processed.images

            image_index = 0
            for _y, _h, row in grid.tiles:
                for tiledata in row:
                    tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                    image_index += 1

            combined_image = images.combine_grid(grid)
            result_images.append(combined_image)
        return result_images

    def poor_mans_outpainting_img2img(self, prompt: str, init_image: Image, pixels: int =128, mask_blur: int =4, inpainting_fill: str ="fill", direction: list = ["left"], negative_prompt: str = "", seed: int = -1, steps: int = 20, cfg_scale: float = 7.0, sampler_name: str = 'Euler', denoising_strength: float = 0.75):
        input_params = {
            'prompt': prompt,
            'init_image': init_image,
            'negative_prompt': negative_prompt,
            'seed': seed,
            'steps': steps,
            'cfg_scale': cfg_scale,
            'sampler_name': sampler_name,
            'denoising_strength': denoising_strength, 
            'mask_blur': mask_blur, 
            'inpainting_fill': inpainting_fill
        }
        input_params = self.__process_args_img2img(**input_params)
        p = StableDiffusionProcessingImg2Img(sd_model=self.__pipe, **input_params)
        p.is_api = True
        p.init_images = [init_image]

        initial_seed = None
        initial_info = None

        p.mask_blur = mask_blur * 2
        p.inpainting_fill = inpainting_fill
        p.inpaint_full_res = False

        left = pixels if "left" in direction else 0
        right = pixels if "right" in direction else 0
        up = pixels if "up" in direction else 0
        down = pixels if "down" in direction else 0

        init_img = p.init_images[0]
        target_w = math.ceil((init_img.width + left + right) / 64) * 64
        target_h = math.ceil((init_img.height + up + down) / 64) * 64

        if left > 0:
            left = left * (target_w - init_img.width) // (left + right)
        if right > 0:
            right = target_w - init_img.width - left

        if up > 0:
            up = up * (target_h - init_img.height) // (up + down)

        if down > 0:
            down = target_h - init_img.height - up

        img = Image.new("RGB", (target_w, target_h))
        img.paste(init_img, (left, up))

        mask = Image.new("L", (img.width, img.height), "white")
        draw = ImageDraw.Draw(mask)
        draw.rectangle((
            left + (mask_blur * 2 if left > 0 else 0),
            up + (mask_blur * 2 if up > 0 else 0),
            mask.width - right - (mask_blur * 2 if right > 0 else 0),
            mask.height - down - (mask_blur * 2 if down > 0 else 0)
        ), fill="black")

        latent_mask = Image.new("L", (img.width, img.height), "white")
        latent_draw = ImageDraw.Draw(latent_mask)
        latent_draw.rectangle((
             left + (mask_blur//2 if left > 0 else 0),
             up + (mask_blur//2 if up > 0 else 0),
             mask.width - right - (mask_blur//2 if right > 0 else 0),
             mask.height - down - (mask_blur//2 if down > 0 else 0)
        ), fill="black")

        devices.torch_gc()

        grid = images.split_grid(img, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_mask = images.split_grid(mask, tile_w=p.width, tile_h=p.height, overlap=pixels)
        grid_latent_mask = images.split_grid(latent_mask, tile_w=p.width, tile_h=p.height, overlap=pixels)

        p.n_iter = 1
        p.batch_size = 1
        p.do_not_save_grid = True
        p.do_not_save_samples = True

        work = []
        work_mask = []
        work_latent_mask = []
        work_results = []

        for (y, h, row), (_, _, row_mask), (_, _, row_latent_mask) in zip(grid.tiles, grid_mask.tiles, grid_latent_mask.tiles):
            for tiledata, tiledata_mask, tiledata_latent_mask in zip(row, row_mask, row_latent_mask):
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                work.append(tiledata[2])
                work_mask.append(tiledata_mask[2])
                work_latent_mask.append(tiledata_latent_mask[2])

        batch_count = len(work)
        print(f"Poor man's outpainting will process a total of {len(work)} images tiled as {len(grid.tiles[0][2])}x{len(grid.tiles)}.")

        for i in range(batch_count):
            p.init_images = [work[i]]
            p.image_mask = work_mask[i]
            p.latent_mask = work_latent_mask[i]

            processed = process_images(p, self.__aliases, self.__model_data, self.__pipe, self.weights_file)

            if initial_seed is None:
                initial_seed = processed.seed

            p.seed = processed.seed + 1
            work_results += processed.images


        image_index = 0
        for y, h, row in grid.tiles:
            for tiledata in row:
                x, w = tiledata[0:2]

                if x >= left and x+w <= img.width - right and y >= up and y+h <= img.height - down:
                    continue

                tiledata[2] = work_results[image_index] if image_index < len(work_results) else Image.new("RGB", (p.width, p.height))
                image_index += 1

        combined_image = images.combine_grid(grid)
        return [combined_image]

def civit_download(url: str, file_save: str):
    civit_model_api = "https://civitai.com/api/v1/models/"
    import requests
    import re

    # Check if the URL format is correct
    model_id_match = re.search(r'models/(\d+)', url)
    if not model_id_match:
        raise ValueError("Invalid URL format for model ID")

    endpoint = civit_model_api + model_id_match.group(1)

    try:
        response = requests.get(endpoint)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Error in HTTP request: {e}")

    model_version_id_match = re.search(r'modelVersionId=(\d+)', url)
    try:
        data = response.json()
    except ValueError:
        raise ValueError("Response content is not valid JSON")

    if model_version_id_match:
        model_version_id = int(model_version_id_match.group(1))
        download_url = None
        for model_version in data.get('modelVersions', []):
            if model_version['id'] == model_version_id:
                download_url = model_version.get('downloadUrl')
                break
        if not download_url:
            raise ValueError("Model version ID not found in the data")
    else:
        download_url = data.get('modelVersions', [{}])[0].get('downloadUrl')

    if not download_url:
        raise ValueError("Download URL not found")

    try:
        response = requests.get(download_url)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(f"Error downloading the file: {e}")

    try:
        with open(file_save, 'wb') as f:
            f.write(response.content)
    except IOError as e:
        raise IOError(f"Error writing file: {e}")
