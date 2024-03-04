import os.path
import sys
import os
import warnings
import torch
# from .EsrganPipelines import EsrganPipeline, RealEsrganPipeline

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
os.environ['TORCH_COMMAND'] = "pip install torch==2.0.1 torchvision==0.15.2"
os.environ['ERROR_REPORTING'] = "FALSE"
os.environ['PIP_IGNORE_INSTALLED'] = "0"
os.environ['SD_WEBUI_RESTART'] = "tmp/restart"

warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision")
warnings.filterwarnings(action="ignore", category=UserWarning, module="torchvision.transforms.functional_tensor")

import io
from PIL import Image, PngImagePlugin, ImageDraw
import base64
import piexif
import math
import random
import importlib

def load_module_at_path(full_path):
    module_name = os.path.basename(full_path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

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
    import base64

    img = cv2.imread(img_path)
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')
    return encoded_image

class ControlNetModel:
    def __init__(self, model_path: str, image: str, module: str = 'none', weight: float = 1.0, 
                 resize_mode: int = 1, lowvram: bool = False, processor_res: int = 512, 
                 threshold_a: int = 1, threshold_b: int = 1, guidance_start: float = 0.0, 
                 guidance_end: float = 1.0, control_mode: int = 0, pixel_perfect: bool = False, 
                 default_command_args = None):

        if not model_path:
            raise ValueError("Parameter 'model_path' is required and cannot be None or empty.")
        if not image:
            raise ValueError("Parameter 'image' is required and cannot be None or empty.")
        
        if default_command_args is None:
            if torch.cuda.is_available():
                os.environ['COMMANDLINE_ARGS'] = "--upcast-sampling --no-half --skip-torch-cuda-test --no-half-vae interrogate"
            elif torch.backends.mps.is_available():
                os.environ['COMMANDLINE_ARGS'] = "--skip-torch-cuda-test --no-half --upcast-sampling --no-half-vae --use-cpu interrogate"
            else:
                os.environ['COMMANDLINE_ARGS'] = "--skip-torch-cuda-test --no-half-vae --no-half interrogate"
        else:
            os.environ['COMMANDLINE_ARGS'] = default_command_args

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
        from ..modules import processing, images, devices, script_callbacks, scripts, extensions
        from ..extensions.controlnet.scripts import global_state

        extensions.list_extensions()
        scripts.load_scripts()

        current_file_path = os.path.dirname(__file__)  # Gets the directory of the current script
        target_module_relative_path = os.path.join(current_file_path, '../extensions/controlnet/scripts/controlnet_ui/controlnet_ui_group.py')

        # Normalize the path to resolve any '..'
        target_module_full_path = os.path.normpath(target_module_relative_path)

        # Load the module
        control_net_ui_group = load_module_at_path(target_module_full_path)
        
        self.config = {
            'enabled': True, 
            'module': module,  # Assuming this remains constant as well
            'model': model_path,
            'weight': weight,
            'image': read_image(image),
            'resize_mode': resize_mode,
            'lowvram': lowvram,
            'processor_res': processor_res,
            'threshold_a': threshold_a,
            'threshold_b': threshold_b,
            'guidance_start': guidance_start,
            'guidance_end': guidance_end,
            'control_mode': control_mode,
            'pixel_perfect': pixel_perfect
        }

        script_runner = scripts.scripts_txt2img

        if not script_runner.scripts:
            script_runner.initialize_scripts(False)

        os.environ['CONTROLNET_MODELS_PATH'] = os.path.dirname(os.path.abspath(model_path))

        print(global_state.cn_models_dir_old)

        # update_cn_models(os.path.dirname(os.path.abspath(model_path)))

        UiControlNetUnit = control_net_ui_group.UiControlNetUnit

        unit_2 = UiControlNetUnit(**UiControlNetUnit_false)
        unit_3 = UiControlNetUnit(**UiControlNetUnit_false)
        controlnet_script_tuple = (self.config, unit_2, unit_3)

        args_list = [controlnet_script_tuple]
        
        current_index = 0
        for script, arguments in zip(script_runner.alwayson_scripts, args_list):
            script.args_from = current_index
            script.args_to = current_index + len(arguments)
            current_index = current_index + len(arguments)

        self.script_runner = script_runner
        self.script_args = controlnet_script_tuple
