# How to run automatic1111sdk on windows, using an nvidia gpu
_by Marco Guardigli, mgua@tomware.it_

auto1111sdk 0.0.93 requires torch==2.1.0
a specific python version between 3.8 and <3.11 is required. 

## python version
on windows, you can have several python versions, installed from python.org. 
specific python environments can be created with a specific python version, using py launcher.

install standard python distributions from python.org, and use the launcher "py" to launch a specific version, before creating the environment. Here are two examples:
	from powershell: 
```
	py -3.8 -m venv venv_python38
	.\venv_python38\Scripts\activate.ps1
	# do whatever you need
	# install packages with python -m pip --verbose install <packagename>
	# use python -V to check the interpreter version
	deactivate

	py -3.11 -m venv venv_python311
	.\venv_python311\Scripts\activate.ps1
	# do whatever you need
	# install packages with python -m pip --verbose install <packagename>
	# use python -V to check the interpreter version
	deactivate
```
Download and install the latest python 3.10.x from python.org. use py -3.10 to create the environment where auto1111sdk will be run.

the following commands will create a new environment and activate it.
```
py -3.10 -m venv venv_auto1111sdk
.\venv_auto1111sdk\Scripts\activate.ps1
```
from now on, within the environment, python will invoke the python 3.10 interpreter

## torch and cuda

Torch is a library from Meta, dedicated to AI. Cuda is a library released by nvidia to allow code to interact with nvidia GPU hardware

Simple torch does not use cuda. A binary torch compiled with cuda libraries is needed. Similarly torchvision benefits from cuda, via a dedicated cuda enabled torchvision package

nvidia gpus allow drivers to be updated at the system level. nvidia runtime is including updated cuda libraries from nvidia. the windows executable nvidia-smi.exe shows system level drivers version and system level cuda version.

current (feb 2024) cuda as reported by my nvidia-smi is 12.2, and this is the system level cuda runtime. You can install a specific level of cuda runtime inside your specific python environment. this runtime is installed from standard repos (no need to go to nvidia repos).
The python specific cuda runtime has to be <= the system level cuda runtime.

python specific cuda runtime 11 install ( https://pypi.org/project/nvidia-cuda-runtime-cu11/ )
```
python -m pip install --verbose nvidia-cuda-runtime-cu11
```
cuda-enabled torch 2.1.0+cu118 and related torchvision
```
python -m pip install --verbose torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```
after this:

```
python
>>> import torch

>>> print("Torch version:",torch.__version__)
Torch version: 2.1.0+cu118

>>> print("Is CUDA enabled?",torch.cuda.is_available())
Is CUDA enabled? True
```

and 
```
python - m pip list | grep torch 
```
can be used to see which cuda levels torch is compiled for

```
python -m pip list | grep torch
open-clip-torch       	2.20.0
pytorch-lightning     	1.9.4
torch                 	2.1.0+cu118
torchdiffeq           	0.2.3
torchmetrics          	1.3.0.post0
torchsde              	0.2.6
torchvision           	0.16.0+cu118
```

## automatic1111sdk

now we can install automatic1111sdk in our environment with
```
python -m pip install --verbose automatic1111sdk
```
and save the package version list in the classical requirements.txt file
```
python -m pip freeze >requirements.txt
```
## putting it all together

Now we can test and run some code: save the following in auto1111sdk_test.py and run it with
python auto1111sdk_test.py

```
#!/usr/bin/python
# see https://github.com/saketh12/Auto1111SDK
'''
Marco Guardigli, mgua@tomware.it

automatic111sdk is a library allowing automatic1111 stablediffusion use from python
                is not relying on gradio webapp.

from a windows powershell:
    py -3.10 -m venv venv_a1111sdk
    .\venv_a1111sdk\scripts\activate.ps1
    python -m pip install pip --upgrade
    python -m pip install --verbose nvidia-cuda-runtime-cu11
    python -m pip install --verbose torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
    python -m pip --verbose install auto1111sdk
    python -m pip freeze >requirements.txt
    
    python thisfile.py

    deactivate


'''

import os

from auto1111sdk import civit_download, download_realesrgan, RealEsrganPipeline, StableDiffusionPipeline, EsrganPipeline
from PIL import Image

import torch
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())

civit_url = 'https://civitai.com/models/4384/dreamshaper'
model_path = 'dreamshaper.safetensors'
if not os.path.exists(model_path):
    print(f'downloading {model_path} from {civit_url}')
    civit_download(civit_url, model_path)
else:
    print(f'using model {model_path}')


print(f'Text to image, model={model_path}')
pipe = StableDiffusionPipeline(model_path)

prompt          = "portrait photo of a beautiful 20 y.o. girl, 8k uhd, high quality, cinematic" #@param{type: 'string'}
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck" #@param{type: 'string'}
print(f'{prompt=}')
print(f'{negative_prompt=}')

num_images      = 1
height          = 768 
width           = 512
steps           = 20 
output_path     = "txt2img.png" 
cfg_scale       = 7.5
seed            = -1 
sampler_name    = 'Euler' 
# ["Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a", 
#  "DPM++ 2S a", "DPM++ 2M", "DPM fast", "DPM adaptive", 
#  "LMS Karras", "DPM2 Karras", "DPM2 a Karras", 
#  "DPM++ 2S a Karras", "DPM++ 2M Karras", "DDIM", "PLMS"]

output = pipe.generate_txt2img(
                    num_images = num_images, cfg_scale = cfg_scale, sampler_name = sampler_name, seed       = seed,
                    prompt     = prompt,     height    = height,    width        = width, 
                    negative_prompt = negative_prompt,              steps        = steps)

output[0].save(output_path)

if os.path.exists(output_path):
    print(f'Text to Image output generated: {output_path}')
else:
    print(f'Error: output file not found {output_path}')

del pipe

```
 
