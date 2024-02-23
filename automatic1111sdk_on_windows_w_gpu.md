# How to run automatic1111sdk on windows, using an nvidia gpu
_by Marco Guardigli, mgua@tomware.it_ 

(updated on feb 23 2024)

auto1111sdk 0.0.93 requires torch==2.1.0

A specific python version between 3.8 and <3.11 is required for torch. 

To enable GPU accelerations, a specific torch compiled with nvidia cuda libraries is required

This document explains how to setup a windows systems and a dedicated python 3.10 environment for auto1111sdk


## python version
On windows, you can have several python versions, installed from python.org. 
Local admin rights are not mandatory to perform the installation.
Specific python environments can be created with a specific python version, using *py* launcher.

We install standard python distributions from python.org, and use the launcher "py" to launch a specific version when creating the environment. 

Here are two examples. Note how the environment creation line uses py to invoke the specific python version.

These commands are to be run from a powershell prompt: 

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

The following commands will create a new environment for auto1111sdk and activate it.

```
py -3.10 -m venv venv_auto1111sdk

.\venv_auto1111sdk\Scripts\activate.ps1
```
Note the prompt change, indicating the new environment is active.

From now on, within the environment, python will invoke the python 3.10 interpreter. 
All the installations and executions for this project will be performed within this python environment.

The following command updates the pip package manager in the newly created environment.

```
python -m pip install pip --upgrade
```
Now the local pip is upgraded. Always invoke pip with python -m pip to be sure 
you run the environment specific pip version.


## torch and cuda

Torch is a library from Meta, dedicated to AI. 

Cuda is a library released by Nvidia to allow code to interact with nvidia GPU hardware.

Simple torch does not use cuda. A binary torch compiled with **specific cuda libraries** is needed. 
Similarly to torch, also the torchvision library benefits from cuda, via a dedicated cuda enabled torchvision package.

Nvidia gpus allow drivers to be updated at the system level. These are updated when needed.
Nvidia runtime comes from Nvidia too, with updated cuda libraries. 

The windows executable **nvidia-smi.exe** shows system level drivers version and system level cuda version.

We do not want that system level cuda libraries, when updated, break our python setup, 
which requires very specific cuda libraries version.

Current (feb 2024) on my machine the cuda version, as reported by nvidia-smi is 12.2, and this is the system level cuda runtime. 

A python dedicated specific level of cuda runtime can be installed within a specific python environment. 
This runtime is installed from standard pip repositories (no need to go to Nvidia repos).

The python specific cuda runtime has to be <= the system level cuda runtime. 
Actually the nvidia-smi tool shows _the highest_ cuda version level that can be supported in current setup 
via the system level components. This is not to be intendend as the only available version.

A suitable cuda version for torch 2.0.1 is 11.8. The corresponding components for python are available to pip via standard repositories. (check https://pypi.org/project/nvidia-cuda-runtime-cu11/ ).

Here we install python nvidia cuda v11 components:
```
python -m pip install --verbose nvidia-cuda-runtime-cu11
```
and then  cuda-enabled torch 2.1.0+cu118 and related torchvision.

It is **important** to install torch and torchvision from the same pip line, otherwise dependencies could break.

Note how we install torch from the specific location (index) dedicated to cuda v11.
(check pytorch.org for details). As you can see, Torch is quite picky about environment releases and dependencies.

```
python -m pip install --verbose torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

After this we can validate torch and cuda availability from our python environment:

```
python
>>> import torch

>>> print("Torch version:",torch.__version__)
Torch version: 2.1.0+cu118

>>> print("Is CUDA enabled?",torch.cuda.is_available())
Is CUDA enabled? True
```

and the following command:
```
python - m pip list | grep torch 
```
can be used to see which cuda levels torch is compiled for. (you need to have grep tool available)

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

Now we are ready to install auto1111sdk in our environment with:
```
python -m pip install --verbose auto1111sdk
```
and we can save the package version list in the classical requirements.txt file
```
python -m pip freeze >requirements.txt
```
to preserve all the dependency choices performed by pip.


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
 
Enjoy!

