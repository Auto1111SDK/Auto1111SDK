import os.path
import sys
import os
import warnings
import torch
from .EsrganPipelines import EsrganPipeline, RealEsrganPipeline

if torch.cuda.is_available():
    os.environ['COMMANDLINE_ARGS'] = "--upcast-sampling --skip-torch-cuda-test --no-half-vae interrogate"
elif torch.backends.mps.is_available():
    os.environ['COMMANDLINE_ARGS'] = "--no-half --api --skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate" #"--no-half --api --skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate"
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
from ..modules import processing, images, devices, script_callbacks, scripts, extensions
import math
import random
import importlib

extensions.list_extensions()
scripts.load_scripts()

def load_module_at_path(full_path):
    module_name = os.path.basename(full_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

current_file_path = os.path.dirname(__file__)  # Gets the directory of the current script
target_module_relative_path = os.path.join(current_file_path, '../extensions/controlnet/scripts/controlnet_ui/controlnet_ui_group.py')

# Normalize the path to resolve any '..'
target_module_full_path = os.path.normpath(target_module_relative_path)

# Load the module
control_net_ui_group = load_module_at_path(target_module_full_path)

def generate_dummy_image(width, height, channels, color=(0, 0, 0)):
    import numpy as np
    """
    Generate a dummy image placeholder with specified dimensions and color.

    Args:
    - width (int): Width of the image.
    - height (int): Height of the image.
    - channels (int): Number of color channels (e.g., 3 for RGB).
    - color (tuple, optional): The color to fill the image with, specified as a tuple
      (R, G, B). Defaults to black (0, 0, 0).

    Returns:
    - numpy.ndarray: A numpy array representing the dummy image.
    """
    # Ensure the color tuple has the same number of elements as there are channels
    if len(color) != channels:
        raise ValueError("Color tuple length must match the number of channels.")
    
    # Create a dummy image filled with the specified color
    image = np.zeros((height, width, channels), dtype=np.uint8)
    if channels > 1:
        for i in range(channels):
            image[:, :, i] = color[i]
    else:
        image[:, :] = color[0]  # Assuming single-channel (grayscale) image

    return image

UiControlNetUnit_false = {
    'enabled': False,
    'module': 'none',
    'model': 'None',
    'weight': 1.0,
    'image': None,
    'resize_mode': 'Crop and Resize',
    'low_vram': False,
    'processor_res': -1,
    'threshold_a': -1,
    'threshold_b': -1,
    'guidance_start': 0.0,
    'guidance_end': 1.0,
    'pixel_perfect': False,
    'control_mode': 'Balanced',
    'inpaint_crop_input_image': False,
    'hr_option': 'Both',
    'save_detected_map': True,
    'advanced_weighting': None
}


def read_image(img_path):
    import cv2

    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image from path: {img_path}")
    
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')
    return encoded_image

UiControlNetUnit_true = {
    "enabled": True,
    "module": None,
    "model": "control_v11p_sd15_lineart",
    "weight": 1,
    "image": {
        "image": read_image('/Users/adityachebrolu/Documents/warpfusion/stock_mountain.png'),
        # "mask": "Placeholder for mask array data"
    },
    "resize_mode": "Crop and Resize",
    "low_vram": False,
    "processor_res": 64,
    "threshold_a": 0,
    "threshold_b": 0,
    "guidance_start": 0,
    "guidance_end": 1,
    "pixel_perfect": False,
    "control_mode": "Balanced",
    "inpaint_crop_input_image": False,
    "hr_option": "Both",
    "save_detected_map": True,
    "advanced_weighting": None
}

config = {
    'enabled': True,
    'module': 'none',
    'model': 'control_v11p_sd15_openpose',
    'weight': 1.0,
    'image': read_image('/Users/adityachebrolu/Documents/warpfusion/stock_mountain.png'),
    'resize_mode': 1,
    'lowvram': False,
    'processor_res': 64,
    'threshold_a': 64,
    'threshold_b': 64,
    'guidance_start': 0.0,
    'guidance_end': 1.0,
    'control_mode': 0,
    'pixel_perfect': False
}

class ControlNetModel:
    def __init__(self):
        script_runner = scripts.scripts_txt2img

        if not script_runner.scripts:
            script_runner.initialize_scripts(False)

        UiControlNetUnit = control_net_ui_group.UiControlNetUnit

        unit_1 = UiControlNetUnit(**UiControlNetUnit_true)
        unit_2 = UiControlNetUnit(**UiControlNetUnit_false)
        unit_3 = UiControlNetUnit(**UiControlNetUnit_false)
        controlnet_script_tuple = (config, unit_2, unit_3)

        args_list = [controlnet_script_tuple]
        
        current_index = 0
        for script, arguments in zip(script_runner.alwayson_scripts, args_list):
            script.args_from = current_index
            script.args_to = current_index + len(arguments)
            current_index = current_index + len(arguments)

        self.script_runner = script_runner
        self.script_args = controlnet_script_tuple

            